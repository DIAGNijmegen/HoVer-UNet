from inference.infer import FastHoVerNetInfer
import argparse

def main(parser):
    args = parser.parse_args()
    images_path = args.images_path
    weights_path = args.weights_path
    save_path = args.save_path
    step = args.step
    ext = args.ext
    overlay = args.overlay
    '''ext = 'png'
    step = 192
    weights_path = '/work/oldWork/default-ubuntu-rinaldi2-work-pvc-215307e1-04be-4abb-8a29-26eb803ad2b3/cristian/weights/distilled-hover-net-pytorch-weights/pytorch_weights_distilled_hovernet_mit_b2_1.pth'
    save_path = '/work/cristian/test_fasthovernet/Infer'
    images_path = '/work/oldWork/default-ubuntu-rinaldi2-work-pvc-215307e1-04be-4abb-8a29-26eb803ad2b3/cristian/data/CoNSeP/Test/Images'''
    fast = FastHoVerNetInfer(
        images_path=images_path,
        weights_path=weights_path,
        save_path=save_path,
        path_size=(256, 256, 3),
        step=step,
        ext=ext,
        overlay=overlay
    )
    fast.infer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="FastHoVerNet inference on tile")

    parser.add_argument("--images_path", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--overlay", type=bool, default=True)
    parser.add_argument("--step", type=int, default=192)
    parser.add_argument("--ext", type=str, default='png')
    main(None)