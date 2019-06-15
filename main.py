from common import args

from models.segnet import SegNet
from models.fcn import Model as FCNet


if args.model == 'segnet':
    model = SegNet()
    if args.test:
        model.test()
    else:
        model.train()

elif args.model == 'fcn':
    model = FCNet(session, network, config["categories"])
