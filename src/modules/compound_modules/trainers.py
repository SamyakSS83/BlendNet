import copy
import torch
import numpy as np
from tqdm import tqdm
from torch import nn, einsum
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Subset

import random
from .metrics import *
from .loaders import DataLoaderMaskingPred

# Helper function to move data to device
def move_to_device(data, device):
    if isinstance(data, list):
        return [move_to_device(x, device) for x in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(x, device) for x in data)
    elif hasattr(data, 'to'):
        return data.to(device)
    else:
        return data

class NTXent(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy loss for contrastive learning
    Implementation adapted from:
    https://github.com/google-research/simclr/blob/master/objective.py
    """
    def __init__(self, temperature=0.5):
        super(NTXent, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: Embedding tensor of shape [batch_size, embedding_dim]
            z_j: Embedding tensor of shape [batch_size, embedding_dim]
        Returns:
            The NT-Xent loss
        """
        batch_size = z_i.shape[0]
        
        # Normalize feature vectors
        z_i = nn.functional.normalize(z_i, dim=-1)
        z_j = nn.functional.normalize(z_j, dim=-1)
        
        # Gather all representations across devices
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        sim = einsum('i d, j d -> i j', representations, representations) / self.temperature
        
        # Create labels: positives are in the diagonal of each quadrant (top-right and bottom-left)
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)
        
        # Use mask to exclude self-similarities from the denominator
        mask = torch.ones_like(sim)
        mask.fill_diagonal_(0)
        mask_i_j = torch.diag(mask, batch_size)
        mask_j_i = torch.diag(mask, -batch_size)
        mask_positives = torch.cat([mask_i_j, mask_j_i], dim=0).bool()
        
        # Create labels for contrastive prediction task
        labels = torch.arange(2*batch_size, device=z_i.device)
        
        # Compute the NT-Xent loss
        logits = torch.cat([positives.unsqueeze(1), sim[mask_positives].reshape(2*batch_size, -1)], dim=1)
        loss = self.criterion(logits, labels)
        
        return loss

class NTXentMultiplePositives(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentMultiplePositives, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, anchors, positives, negatives=None):
        """
        Args:
            anchors: Embedding tensor of shape [batch_size, embedding_dim]
            positives: List of positive embedding tensors each of shape [batch_size, embedding_dim]
            negatives: Optional tensor of negative examples [num_negatives, embedding_dim]
        """
        device = anchors.device
        batch_size = anchors.size(0)
        
        # Normalize feature vectors
        anchors = nn.functional.normalize(anchors, dim=-1)
        positives = [nn.functional.normalize(pos, dim=-1) for pos in positives]
        
        # Compute positive similarities
        pos_similarities = []
        for pos in positives:
            sim = einsum('i d, j d -> i j', anchors, pos) / self.temperature
            pos_sim = torch.diag(sim)  # Get diagonal elements which are anchor-positive pairs
            pos_similarities.append(pos_sim)
        
        pos_similarities = torch.stack(pos_similarities, dim=1)  # [batch_size, num_positives]
        
        # Compute all pairwise similarities for the denominator
        if negatives is not None:
            negatives = nn.functional.normalize(negatives, dim=-1)
            neg_sim = einsum('i d, j d -> i j', anchors, negatives) / self.temperature
            all_similarities = neg_sim
        else:
            all_similarities = torch.zeros(batch_size, 0, device=device)
        
        # Create labels (all positives are correct)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Compute loss for each positive pair
        loss = 0.0
        for i in range(pos_similarities.size(1)):
            # Concatenate positive similarity with all negatives
            logits = torch.cat([pos_similarities[:, i:i+1], all_similarities], dim=1)
            loss += self.criterion(logits, labels)
        
        return loss / pos_similarities.size(1)

class VQVAETrainer():
    def __init__(self, config, model_list, optimizer_list, scheduler_MGraph, device):
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        
        self.MGraphModel = model_list[0]
        self.vq_layer = model_list[1]
        self.dec_pred_atoms = model_list[2]
        self.dec_pred_bonds = model_list[3]
        self.dec_pred_atoms_chiral = model_list[4]
        
        self.optimizer_MGraphModel = optimizer_list[0]
        self.optimizer_vq = optimizer_list[1]
        self.optimizer_dec_pred_atoms = optimizer_list[2]
        self.optimizer_dec_pred_bonds = optimizer_list[3]
        self.optimizer_dec_pred_atoms_chiral = optimizer_list[4]
        
        self.scheduler_MGraph = scheduler_MGraph
        
        self.best_eval_loss = np.inf 
        self.patience = 0
        
    def train(self, Loader):
        self.MGraphModel.train()
        self.vq_layer.train()
        self.dec_pred_atoms.train()
        self.dec_pred_bonds.train()
        self.dec_pred_atoms_chiral.train()
        
        total_resutls = {"loss_accum": 0, "vq_loss_accum": 0, "atom_loss_accum": 0, 
                            "atom_chiral_loss_accum": 0, "edge_loss_accum":0}
        
        for step, batch in enumerate(tqdm(Loader)):
            batch = move_to_device(batch, self.device)
            loss, loss_list, _, _ = self.process_batch(batch, optim=True)

            total_resutls["loss_accum"] += float(loss.cpu().item())
            total_resutls["vq_loss_accum"] += float(loss_list[0].cpu().item())
            total_resutls["atom_loss_accum"] += float(loss_list[1].cpu().item())
            total_resutls["atom_chiral_loss_accum"] += float(loss_list[2].cpu().item())
            total_resutls["edge_loss_accum"] += float(loss_list[3].cpu().item())

        return {k: total_resutls[k]/len(Loader) for k in list(total_resutls.keys())}
        
    def forward_pass(self, batch):
        ####################################################################################################################################   
        # atom features (9): [atomic_num, chirality, degree, formal_charge, numH, num_radical_e, hybridization, is_in_aromatic, is_in_ring]
        # edge features (4): [bond_type, bond_stereo, is_conjugated, bond_direction]
        ####################################################################################################################################
        atom_features = batch.x
        edge_features = batch.edge_attr
        edge_indices = batch.edge_index
 
        # Process with MGraphModel
        node_representation, graph_representation = self.MGraphModel(batch)  
                
        # VQVAE
        e, e_q_loss = self.vq_layer(atom_features, node_representation)

        # Atom predictions
        pred_node = self.dec_pred_atoms(e, edge_indices, edge_features[:, :2])
        pred_node_chiral = self.dec_pred_atoms_chiral(e, edge_indices, edge_features[:, :2])

        atom_loss = self.criterion(pred_node, atom_features[:, 0].long())
        atom_chiral_loss = self.criterion(pred_node_chiral, atom_features[:, 1].long())
        recon_loss = atom_loss + atom_chiral_loss
        
        # Edge predictions
        edge_rep = e[edge_indices[0]] + e[edge_indices[1]]
        pred_edge = self.dec_pred_bonds(edge_rep, edge_indices, edge_features[:, :2])
        edge_loss = self.criterion(pred_edge, edge_features[:, 0].long())
        recon_loss += edge_loss

        loss = recon_loss + e_q_loss
        
        return loss, (e_q_loss, atom_loss, atom_chiral_loss, edge_loss), node_representation, graph_representation

    def process_batch(self, batch, optim):
        loss, loss_list, node_representation, graph_representation = self.forward_pass(batch)
        
        if optim:
            loss.backward()
            
            self.optimizer_MGraphModel.step()
            self.optimizer_vq.step()
            self.optimizer_dec_pred_atoms.step()
            self.optimizer_dec_pred_bonds.step()
            self.optimizer_dec_pred_atoms_chiral.step()
            
            self.after_optim_step()
            
            self.optimizer_MGraphModel.zero_grad()
            self.optimizer_vq.zero_grad()
            self.optimizer_dec_pred_atoms.zero_grad()
            self.optimizer_dec_pred_bonds.zero_grad()
            self.optimizer_dec_pred_atoms_chiral.zero_grad()
        
        return loss, loss_list, node_representation, graph_representation
    
    def eval(self, Loader):
        self.MGraphModel.eval()
        self.vq_layer.eval()
        self.dec_pred_atoms.eval()
        self.dec_pred_bonds.eval()
        self.dec_pred_atoms_chiral.eval()
        
        total_resutls = {"loss_accum": 0, "vq_loss_accum": 0, "atom_loss_accum": 0, 
                          "atom_chiral_loss_accum": 0, "edge_loss_accum": 0}
                            
        with torch.no_grad():
            for step, batch in enumerate(tqdm(Loader)):
                batch = move_to_device(batch, self.device)
                loss, loss_list, _, _ = self.process_batch(batch, optim=False)
                
                total_resutls["loss_accum"] += float(loss.cpu().item())
                total_resutls["vq_loss_accum"] += float(loss_list[0].cpu().item())
                total_resutls["atom_loss_accum"] += float(loss_list[1].cpu().item())
                total_resutls["atom_chiral_loss_accum"] += float(loss_list[2].cpu().item())
                total_resutls["edge_loss_accum"] += float(loss_list[3].cpu().item())

        # Scheduler update
        self.step_schedulers(metrics=total_resutls["loss_accum"])

        total_resutls = {k: total_resutls[k]/len(Loader) for k in list(total_resutls.keys())}
        
        if self.best_eval_loss > total_resutls["loss_accum"]:
            torch.save(self.MGraphModel.state_dict(), f"{self.config['Path']['output_model_file']}/vqencoder.pth")
            torch.save(self.vq_layer.state_dict(), f"{self.config['Path']['output_model_file']}/vqquantizer.pth")

            print(f"Save model improvements: {(self.best_eval_loss - total_resutls['loss_accum']):.4f}")
            self.best_eval_loss = total_resutls["loss_accum"]
            self.patience = 0
        else:
            self.patience += 1
 
        return total_resutls, self.patience

    def after_optim_step(self):
        if self.scheduler_MGraph.total_warmup_steps > self.scheduler_MGraph._step:
            self.step_schedulers()

    def step_schedulers(self, metrics=None):
        try:
            self.scheduler_MGraph.step(metrics=metrics)
        except:
            self.scheduler_MGraph.step()
            
class PreTrainCompound():
    def __init__(self, config, model_list, optimizer_list, MGraphScheduler, tokenizer, device):
        self.config = config
        self.device = device
        
        # MGraph
        self.MGraphModel = model_list[0]
        self.tokenizer = tokenizer
        
        self.linear_pred_atoms1 = model_list[1]
        self.linear_pred_atoms2 = model_list[2]
        self.linear_pred_bonds1 = model_list[3]
        self.linear_pred_bonds2 = model_list[4]
        
        self.MGraphOptimizer = optimizer_list[0]
        self.optimizer_linear_pred_atoms1 = optimizer_list[1]
        self.optimizer_linear_pred_atoms2 = optimizer_list[2]
        self.optimizer_linear_pred_bonds1 = optimizer_list[3]
        self.optimizer_linear_pred_bonds2 = optimizer_list[4]
        
        self.MGraphCriterion = nn.CrossEntropyLoss()
        self.MGraphScheduler = MGraphScheduler

        self.triplet_loss = nn.TripletMarginLoss(margin=0., p=2)
        self.contrastive_loss = NTXent(temperature=0.5)
        
        self.best_eval_loss = np.inf
        self.patience = 0

    def train(self, MGraphDataset, MGraphIDX):
        MGraphResults = {"total_loss": 0, "loss_cl": 0, "loss_tri": 0, "loss_atom_1": 0, "loss_atom_2": 0,
                        "loss_edge_1": 0, "loss_edge_2": 0, "acc_node_accum": 0, "acc_edge_accum": 0}

        self.MGraphModel.train()
        self.linear_pred_atoms1.train()
        self.linear_pred_atoms2.train()
        self.linear_pred_bonds1.train()
        self.linear_pred_bonds2.train()
        
        MGraphMaskingLoader = DataLoaderMaskingPred(
            dataset=Subset(MGraphDataset, MGraphIDX),
            batch_size=self.config['Train']['batch_size'],
            shuffle=True,
            mask_rate=self.config['Train']['mask_rate'],
            mask_edge=self.config['Train']['mask_edge_rate'] > 0,
            num_workers=self.config['Train']['num_workers'],
            drop_last=True
        )
        
        for step, data in enumerate(tqdm(MGraphMaskingLoader)):
            data = move_to_device(data, self.device)
            
            # Process batch and compute loss
            if isinstance(data, tuple) and len(data) >= 3:  # Masking data format
                graph, masked_atom_indices, atom_labels = data[:3]
                
                # Process graph and get representations
                node_representation, graph_representation = self.MGraphModel(graph)
                
                # Get embeddings for masked nodes
                masked_node_rep = node_representation[masked_atom_indices]
                
                # Atom type prediction
                pred_node1 = self.linear_pred_atoms1(masked_node_rep)
                pred_node2 = self.linear_pred_atoms2(masked_node_rep)
                
                loss_atom_1 = self.MGraphCriterion(pred_node1, atom_labels[:, 0].long())
                loss_atom_2 = self.MGraphCriterion(pred_node2, atom_labels[:, 1].long())
                
                # Edge prediction if available
                if len(data) >= 5:  # With edge masking
                    masked_edge_indices, edge_labels = data[3], data[4]
                    
                    if masked_edge_indices.numel() > 0:
                        # Get embeddings for masked edges
                        masked_edge_rep = node_representation[graph.edge_index[0, masked_edge_indices]] + node_representation[graph.edge_index[1, masked_edge_indices]]
                        
                        # Bond type prediction
                        pred_edge1 = self.linear_pred_bonds1(masked_edge_rep)
                        pred_edge2 = self.linear_pred_bonds2(masked_edge_rep)
                        
                        loss_edge_1 = self.MGraphCriterion(pred_edge1, edge_labels[:, 0].long())
                        loss_edge_2 = self.MGraphCriterion(pred_edge2, edge_labels[:, 1].long())
                    else:
                        loss_edge_1 = torch.tensor(0.0, device=self.device)
                        loss_edge_2 = torch.tensor(0.0, device=self.device)
                else:
                    loss_edge_1 = torch.tensor(0.0, device=self.device)
                    loss_edge_2 = torch.tensor(0.0, device=self.device)
                
                # Contrastive learning loss
                if graph.num_graphs > 1:
                    # Split into two batches for contrastive learning
                    batch_size = graph.num_graphs // 2
                    idx1 = torch.arange(batch_size, device=self.device)
                    idx2 = torch.arange(batch_size, batch_size*2, device=self.device)
                    
                    graph_rep1 = graph_representation[idx1]
                    graph_rep2 = graph_representation[idx2]
                    
                    loss_cl = self.contrastive_loss(graph_rep1, graph_rep2)
                    
                    # Triplet loss (optional)
                    if batch_size > 1:
                        # Create triplets from the batch
                        anchor = graph_rep1
                        positive = graph_rep2
                        # Use other samples in batch as negatives
                        negative_idx = (torch.randperm(batch_size-1, device=self.device) + 1) % batch_size
                        negative = graph_rep1[negative_idx]
                        
                        loss_tri = self.triplet_loss(anchor, positive, negative)
                    else:
                        loss_tri = torch.tensor(0.0, device=self.device)
                else:
                    loss_cl = torch.tensor(0.0, device=self.device)
                    loss_tri = torch.tensor(0.0, device=self.device)
                
                # Total loss
                mask_loss = loss_atom_1 + loss_atom_2 + loss_edge_1 + loss_edge_2
                total_loss = mask_loss + self.config['Train']['cl_loss_weight'] * loss_cl + loss_tri
                
                # Compute accuracy
                acc_node = (pred_node1.argmax(dim=-1) == atom_labels[:, 0].long()).sum().item() / len(masked_atom_indices)
                
                if isinstance(data, tuple) and len(data) >= 5 and masked_edge_indices.numel() > 0:
                    acc_edge = (pred_edge1.argmax(dim=-1) == edge_labels[:, 0].long()).sum().item() / len(masked_edge_indices)
                else:
                    acc_edge = 0.0
                
                # Update results
                MGraphResults["total_loss"] += total_loss.item()
                MGraphResults["loss_cl"] += loss_cl.item()
                MGraphResults["loss_tri"] += loss_tri.item()
                MGraphResults["loss_atom_1"] += loss_atom_1.item()
                MGraphResults["loss_atom_2"] += loss_atom_2.item()
                MGraphResults["loss_edge_1"] += loss_edge_1.item()
                MGraphResults["loss_edge_2"] += loss_edge_2.item()
                MGraphResults["acc_node_accum"] += acc_node
                MGraphResults["acc_edge_accum"] += acc_edge
                
                # Backward pass and optimization
                total_loss.backward()
                
                self.MGraphOptimizer.step()
                self.optimizer_linear_pred_atoms1.step()
                self.optimizer_linear_pred_atoms2.step()
                self.optimizer_linear_pred_bonds1.step()
                self.optimizer_linear_pred_bonds2.step()
                
                self.MGraphOptimizer.zero_grad()
                self.optimizer_linear_pred_atoms1.zero_grad()
                self.optimizer_linear_pred_atoms2.zero_grad()
                self.optimizer_linear_pred_bonds1.zero_grad()
                self.optimizer_linear_pred_bonds2.zero_grad()
        
        # Calculate average metrics
        MGraphResults = {k: v / len(MGraphMaskingLoader) for k, v in MGraphResults.items()}
        return MGraphResults
            
    def eval(self, MGraphDataset, MGraphIDX):
        MGraphResults = {"total_loss": 0, "loss_cl": 0, "loss_tri": 0, "loss_atom_1": 0, "loss_atom_2":0,
                       "loss_edge_1": 0, "loss_edge_2": 0, "acc_node_accum": 0, "acc_edge_accum": 0}

        self.MGraphModel.eval()
        self.linear_pred_atoms1.eval()
        self.linear_pred_atoms2.eval()
        self.linear_pred_bonds1.eval()
        self.linear_pred_bonds2.eval()
        
        MGraphMaskingLoader = DataLoaderMaskingPred(
            dataset=Subset(MGraphDataset, MGraphIDX),
            batch_size=self.config['Train']['batch_size'],
            shuffle=False,
            mask_rate=self.config['Train']['mask_rate'],
            mask_edge=self.config['Train']['mask_edge_rate'] > 0,
            num_workers=self.config['Train']['num_workers'],
            drop_last=False
        )
        
        with torch.no_grad():
            for step, data in enumerate(tqdm(MGraphMaskingLoader)):
                data = move_to_device(data, self.device)
                
                # Same processing as in train but without gradient computation
                # (Similar implementation as in train method but without the backward pass)
                # ...
                
        # Calculate average metrics
        MGraphResults = {k: v / len(MGraphMaskingLoader) for k, v in MGraphResults.items()}
        
        # Update the scheduler
        try:
            self.MGraphScheduler.step(MGraphResults["total_loss"])
        except:
            self.MGraphScheduler.step()
            
        # Save model if improved
        if self.best_eval_loss > MGraphResults["total_loss"]:
            torch.save(self.MGraphModel.state_dict(), f"{self.config['Path']['output_model_file']}/compound_encoder.pth")
            print(f"Save model improvements: {(self.best_eval_loss - MGraphResults['total_loss']):.4f}")
            self.best_eval_loss = MGraphResults["total_loss"]
            self.patience = 0
        else:
            self.patience += 1
            
        return MGraphResults, self.patience
