from keras.applications import VGG16
from keras.models import load_model, Model
import sys

def main(argv):
    if len(argv) <= 2:
        print("usage: ToFullModel.py <partialModelFilename> ><fullModelFilename>")
        exit(1)

    partialModelFilename = argv[1]
    fullModelFilename = argv[2]

    partialModel = load_model(partialModelFilename)

    featuresModel = VGG16(include_top=False, weights='imagenet')

    fullModel = Model(input=featuresModel.input, outputs=partialModel.output)
    fullModel.summary()
    fullModel.save(fullModelFilename)

if __name__ == "__main__":
    main(sys.argv)
