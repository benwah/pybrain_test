from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
import PIL
import os
import itertools
import pickle


THIS_DIR = os.path.dirname(os.path.realpath(__file__))
IMAGE_DIRS = {
    'with_text': os.path.join(THIS_DIR, 'images', 'with_text'),
    'no_text': os.path.join(THIS_DIR, 'images', 'no_text'),
    'test': os.path.join(THIS_DIR, 'images', 'test'),
}
INPUT_MATRIX_SIZE = (200, 200)
MODE = 'RGB'


def get_pixels_from_image(image_path):
    pixels = []
    img = PIL.Image.open(image_path)
    img.convert(MODE)
    img.thumbnail(INPUT_MATRIX_SIZE, PIL.Image.ANTIALIAS)
    dest_matrix = PIL.Image.new(MODE, INPUT_MATRIX_SIZE)
    dest_matrix.paste(img, (0, 0))
    for px in itertools.product(
            xrange(0, dest_matrix.size[0]),
            xrange(0, dest_matrix.size[1])):
        for rgb in dest_matrix.getpixel(px):
            pixels.append(rgb / 255.0)
    return pixels


def test_image(imgname, net, base_path=IMAGE_DIRS['test']):
    image_path = os.path.join(base_path, imgname)
    res = net.activate(get_pixels_from_image(image_path))
    if res < 0:
        print "Not text!", res
    elif res > 0:
        print "Has text!", res
    else:
        print "Not sure.."

    PIL.Image.open(image_path).show()


def save_net(net, filename):
    f = open(filename, 'wb')
    f.write(pickle.dumps(net))
    f.close()


def open_net(filename):
    return pickle.loads(open(filename, 'rb').read())


def main():
    # import ipdb; ipdb.set_trace()
    input_size = INPUT_MATRIX_SIZE[0] * INPUT_MATRIX_SIZE[1] * 3

    net = buildNetwork(
        input_size,
        5 * 5,
        5 * 5,
        1,
        bias=True)

    ds = SupervisedDataSet(input_size, 1)

    for img in os.listdir(IMAGE_DIRS['with_text']):
        ds.addSample(
            get_pixels_from_image(os.path.join(IMAGE_DIRS['with_text'], img)),
            [1.0])

    for img in os.listdir(IMAGE_DIRS['no_text']):
        ds.addSample(
            get_pixels_from_image(os.path.join(IMAGE_DIRS['no_text'], img)),
            [-1.0])

    trainer = BackpropTrainer(net, ds)

    for a in xrange(0, 40):
        print "Training..", a, trainer.train()

    for img in os.listdir(IMAGE_DIRS['test']):
        test_image(img, net)
        import ipdb; ipdb.set_trace()
        
    # base_path=IMAGE_DIRS['test']
    # for pixels in read_images('with_text'):
    #     print "Results: ", net.activate(pixels)

    # for pixels in read_images('no_text'):
    #     print "Results: ", net.activate(pixels)



if __name__ == '__main__':
    main()
