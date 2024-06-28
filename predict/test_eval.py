from mmdet.apis import DetInferencer

# Setup a checkpoint file to load
checkpoint = '/root/autodl-tmp/pannuke_app/predict/maskrcnn/epoch_1.pth'
config_path = "/root/autodl-tmp/pannuke_app/predict/maskrcnn/consep_config.py"
# Initialize the DetInferencer
inferencer = DetInferencer(model=config_path, weights=checkpoint, device='cpu')

result = inferencer("/root/processed/CoNSeP/test/imgs",
                    no_save_pred = False,
                    out_dir='./',
                    return_datasample=True,
          )