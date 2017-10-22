import coremltools
import sys

def main(argv):
    if len(argv) <= 2:
        print("usage: ConvertCoreMLTools.py <keras filename> <mlmodel filename>")
        exit(1)
    kerasFilename = argv[1]
    mlmodelFilename = argv[2]
    
    coreml_model = coremltools.converters.keras.convert(kerasFilename, input_names="image", image_input_names="image", class_labels=["man", "woman"])
    coreml_model.save(mlmodelFilename)

if __name__ == "__main__":
    main(sys.argv)
