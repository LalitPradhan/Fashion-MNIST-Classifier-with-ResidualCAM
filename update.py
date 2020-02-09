from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2, torch
import PIL.ImageOps

# generate class activation mapping for the top1 prediction
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def get_cam(net, features_blobs, img_pil, classes, imgCopy, useModel, device):
    img_pil = PIL.ImageOps.invert(img_pil)
    params = list(net.parameters())
    if useModel == 'CustomNet2':
        weight_softmax = np.squeeze(params[-6].data.cpu().numpy())
        preprocess = transforms.Compose([transforms.Resize((28,28), interpolation=2),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        img_tensor = preprocess(img_pil)
        img_variable = Variable(img_tensor.unsqueeze(0)).to(device)
    elif useModel == 'CustomVGG16_BN':
        weight_softmax = np.squeeze(params[-8].data.cpu().numpy())
        preprocess = transforms.Compose([transforms.Resize((224,224), interpolation=2),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        img_tensor = preprocess(img_pil)
        img_variable = Variable(img_tensor.unsqueeze(0)).to(device)
        img_variable = torch.squeeze(torch.stack((img_variable,img_variable,img_variable),2),0)

    
    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # output: the prediction
    # for i in range(0, 2):
    #     line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
    #     print(line)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0].item()])
    img = np.asarray(img_pil)
    height, width= img.shape
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.3 +imgCopy * 0.5
    result = result.astype(np.uint8)
    # cv2.imshow('frame',heatmap)
    # cv2.imwrite('cam.jpg', result)
    # import pdb;pdb.set_trace()
    return result
