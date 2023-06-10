import torch
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops


class Ensemble:
    """Ensemble of yolov8 models"""

    def __init__(self, models, iou=0.5, conf=0.5, max_det=300):
        self.models = models
        self.max_det = max_det
        self.retina_masks = False
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        for m in self.models:
            m.model.to(self.device)

    def preprocess(self, img):
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1).unsqueeze(0)
        img = img.to(self.device).float()
        img /= 255.0
        return img

    def predict(self, image, augment=False, conf=0.5, iou=0.5):
        # Expect all model use sample image shape
        im = self.preprocess(image)
        # print(len(self.models[0].model(im, augment)))
        # raise
        preds, protos = [], []
        for model in self.models:
            p = model.model(im, augment)
            preds.append(p[0])
            protos.append(p[1][-1])
    
        preds = torch.cat(preds, 2)  # nms ensemble, y shape(B, HW, C)
        proto = torch.stack(protos).mean(0)
        predictions = self.postprocess(preds, proto, im, image,  conf=conf, iou=iou)
        return predictions

    def postprocess(self, preds, proto, img, orig_imgs, conf=0.5, iou=0.5):
            """TODO: filter by classes."""
            p = ops.non_max_suppression(preds,
                                        conf,
                                        iou,
                                        agnostic=False,
                                        max_det=self.max_det,
                                        nc=len(self.models[0].names),
                                        classes=None)
            results = []
            # proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
            for i, pred in enumerate(p):
                orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
                if not len(pred):  # save empty boxes
                    results.append(Results(orig_img=orig_img, path=None, names=self.models[0].names, boxes=pred[:, :6]))
                    continue
                if self.retina_masks:
                    if not isinstance(orig_imgs, torch.Tensor):
                        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                    masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
                else:
                    masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                    if not isinstance(orig_imgs, torch.Tensor):
                        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
                results.append(
                    Results(orig_img=orig_img, path=None, names=self.models[0].names, boxes=pred[:, :6], masks=masks))
            return results
