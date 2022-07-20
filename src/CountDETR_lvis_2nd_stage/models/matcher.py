# Building Hungarian Matcher
# Borrow code from AnchorDETR
# We replace bounding box matching with point location matching

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from util.box_ops import generalized_box_iou, box_cxcywh_to_xyxy

class HungarianMatcher(nn.Module):

    def __init__(self,
                 cost_class: float = 1.,
                 cost_points: float = 1.):
        
        """Create the matcher
        Params:
        cost_class: Class weight
        cost_dists: distance weight
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_points = cost_points
    
    def forward(self,outputs,targets):
        """Matching pipeline

        Args:
            outputs (dict): contains at least two params:
            pred_logits: [batch_size, num_queries, num_classes]: classification logits
            pred_points: [batch_size, num_queries, 2]: predicted points
            targets (list of targets, where len(targets) = batch_size), each target is a dict containing
            labels: tensor of dim [num_target_boxes] containing the class label
            points: tensor of dim [num_target_boxes,2]: target points coordinate
        
        Returns:
            A list of size batch_size, containing the tuple of (index_i, index_j) where:
             - index_i: index of selected predictions (in order)
             - index_j: index of corresponding selected targets

        """
        with torch.no_grad():
            bs, num_queries = outputs['pred_logits'].shape[:2]
            # Flatten to compute cost matrix of the batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_points = outputs["pred_points"].flatten(0,1) #[batch_size * num_queries, 2]

            # Also concat target labels and points
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_points = torch.cat([v["points"] for v in targets]) #[batch_size*num_targets,2]
            # Compute the classification loss
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids] #[num_queries, num_targets]
            # L1 loss
            cost_points = torch.cdist(out_points, tgt_points, p = 1)
            # Add cost
            C = self.cost_class*cost_class + self.cost_points*cost_points
            C = C.view(bs,num_queries,-1).cpu()

            sizes = [len(v["points"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class PointsDistance(nn.Module):
    def __init__(self,dist_type):

        '''
        Accept two distance type: EMD and Chamfer
        '''
        super().__init__()
        self.dist_type = dist_type
    
    def _get_src_permutation_idx(self,indices):

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def em_distance(self,outputs,targets):
        with torch.no_grad():
            bs, num_queries = outputs['pred_points'].shape[:2]
            out_points = outputs['pred_points'].flatten(0,1) #[batch_size * numqueries,2]
            tgt_points = torch.cat([v['points'] for v in targets]) #[batch_size * num_targets,2]
            C = torch.norm(out_points[:,None,:] - tgt_points[None,:,:],p=2,dim=-1) #[batch_size*num_queries,batch_size*num_targets]
            C = C.view(bs,num_queries,-1).cpu()
            sizes = [len(v["points"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
        
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_points'][idx]
        tgt_points = torch.cat([t['points'][i] for t, (_,i) in zip(targets,indices)])

        dists = torch.norm(src_points-tgt_points,p=2,dim=-1)
        return torch.mean(dists), indices
    
    def chamfer_distance(self,outputs,targets):
        with torch.no_grad():
            bs, num_queries = outputs['pred_points'].shape[:2]
            out_points = outputs['pred_points'].flatten(0,1) #[batch_size * num_queries,2]
            tgt_points = torch.cat([v['points'] for v in targets]) #[batch_size * num_targets,2]
            C = torch.norm(out_points[:,None,:] - tgt_points[None,:,:],p=2,dim=-1) #[batch_size * num_queries, batch_size * num_targets]
            C = C.view(bs,num_queries,-1) # [batch_size, num queries, num_targets]

            indices_src = torch.argmin(C,dim=1)
            indices_tgt = torch.argmin(C,dim=2)

        src_points = outputs['pred_points']
        tgt_points = torch.stack([v['points'] for v in targets])
        matched_src = tgt_points[torch.arange(indices_tgt.shape[0]),torch.reshape(indices_tgt,[-1])]
        matched_tgt = src_points[torch.arange(indices_src.shape[0]),torch.reshape(indices_src,[-1])]
    
        src_points = src_points.flatten(0,1)
        tgt_points = tgt_points.flatten(0,1)

        chamfer_dist = torch.mean(torch.norm(src_points - matched_src,p=2,dim=-1)) + \
                       torch.mean(torch.norm(matched_tgt - tgt_points,p=2,dim=-1))
        
        return chamfer_dist, indices_src
    
    def forward(self, outputs, targets):
        if self.dist_type == 'emd':
            return self.em_distance(outputs,targets)
        elif self.dist_type == 'chamfer':
            return self.chamfer_distance(outputs,targets)
        
        else:
            raise NotImplementedError("not support other distance")

class ChamferDistanceMatching(nn.Module):
    def __init__(self, point_cost, giou_cost):
        super().__init__()
        self.point_cost = point_cost
        self.giou_cost = giou_cost

    def forward(self,outputs, targets):
        '''
        Expected parameters in the form
        dictionary, expected in the form:
        pred_boxes: [l,t,r,b]: the bounding position corresponds to anchor position
        points: [x,y]: coordinates of each anchor points
        targets: list of dictionary
        boxes: [cx,cy,w,h]: target bounding boxes
        '''
        with torch.no_grad():
            bs, num_queries = outputs['pred_boxes'].shape[:2]
            out_boxes = outputs['pred_boxes'].flatten(0,1) #[batch_size*num_queries,4]
            tgt_boxes = torch.cat([v['boxes'] for v in targets]) # [batch_size * num_targets,4]

            cost_points = torch.cdist(out_boxes[...,:2],tgt_boxes[...,:2]) #[batch_size*num_queries,batch_size*num_targets]
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_boxes),
                                             box_cxcywh_to_xyxy(tgt_boxes))
            
            C = self.point_cost*cost_points + self.giou_cost*cost_giou

            C = C.view(bs,num_queries,-1).cpu()
            
            indices_src = torch.argmin(C,dim=1)
            indices_tgt = torch.argmin(C,dim=2)
        
        return indices_src, indices_tgt

class OriginalHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
def build_matcher(args):
    return OriginalHungarianMatcher(args.cost_class,
                                    args.cost_bbox,
                                    args.cost_giou)


def build_chamfer_matcher(args):
    return ChamferDistanceMatching(args.chamfer_point_cost,args.chamfer_giou_cost)



class PointHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_point: float = 1,):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_point: This is the relative weight of the L1 error of the point in the matching cost
        """
        super().__init__()
        self.cost_point = cost_point
        assert cost_point != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            # We flatten to compute the cost matrices in a batch
            out_point = outputs["pred_points"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target point
            tgt_point = torch.cat([v["points"] for v in targets])

            # Compute the L1 cost between points
            cost_point = torch.cdist(out_point, tgt_point, p=1)

            # Final cost matrix
            C = self.cost_point * cost_point
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_centerness_matcher(args):
    return PointHungarianMatcher(cost_point=args.set_cost_points)