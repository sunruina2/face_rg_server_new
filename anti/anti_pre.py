import torch
import cv2
from anti.models import encoders
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class AntiSpoofing():
    def __init__(self, img_size=(256, 256), model_path='epoch_best_contr.pth', Threshold=0.02, stretch=False, device='cpu'):
        super(AntiSpoofing, self).__init__()
        self.img_size = img_size

        exe_path = os.path.abspath(__file__)
        # fontpath = str(exe_path.split('face_rg_server/')[0]) + 'face_rg_server/' + "data_pro/wryh.ttf"
        model_path = '../face_rg_files/premodels/pm_anti/' + model_path
        self.model_path = model_path
        self.stretch = stretch
        self.Threshold = Threshold
        self.device = device

    def load_model(self, encoder='se_resnext50'):
        model = getattr(encoders, encoder)(device=self.device,
                                           out_features=1,
                                           pretrained=False)
        checkpoint = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

    def __call__(self, img):
        img = cv2.resize(img, self.img_size) / 255

        assert img.shape[:2] == self.img_size
        assert img.max() <= 1
        image = torch.tensor(img).float().view(1, -1, 256, 256)
        model = self.load_model()
        with torch.no_grad():
            pred, pred_D, _ = model(image)
            probability = torch.sigmoid(pred.view(-1)).numpy()[0]
            if probability > self.Threshold:
                return 1
            else:
                return 0


if __name__ == '__main__':
    anti = AntiSpoofing()
    image = cv2.imread('WechatIMG340.jpeg', cv2.IMREAD_COLOR)
    result = anti(image)
    print(result)
