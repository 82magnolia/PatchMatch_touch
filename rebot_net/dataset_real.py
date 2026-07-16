import os

from dataset import TactileTransferDataset


class RealTactileTransferDataset(TactileTransferDataset):
    """Paired transferred/GT tactile videos from the real-data transfer pipeline.

    Same (transferred → query) supervision as TactileTransferDataset, but the
    real-data pipeline (e.g. transfer_pipeline_real_data_gt_retrieval_*) nests
    each object's videos one level deeper and uses odd query indices instead of
    the synthetic 0-7 numbering:

        transfer_dir/
            {obj_id}/
                transfer/
                    {pair_idx}_transferred.mp4   # network input, blank source
                    {pair_idx}_query_shadow.mp4  # ground truth
                    {pair_idx}_ref_shadow.mp4    # reference (viz only)
                odd_to_even.tsv
                retrieval/

    Only the '_transferred' (dinov3_feat_match backend) naming is produced here;
    _resolve_lq_path handles it. Pair indices are the odd queries 1,3,5,7,9 and
    some objects are missing individual pairs, so NUM_PAIRS spans 0..9 and the
    existence check in the base class skips whatever is absent.
    """

    NUM_PAIRS = 10  # real-data query indices are odd, up to 9

    def _obj_dir(self, obj_id):
        return os.path.join(self.transfer_dir, str(obj_id), 'transfer')
