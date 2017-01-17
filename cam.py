from keras.models import *
from keras.callbacks import *
import keras.backend as K
from model import *
from data import *
from quiver_engine import server
import cv2
import argparse

def train(dataset_path, model_path):
  with K.tf.device('/cpu'):
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))
        if(model_path):
          print "Loading model from path"
          model = load_model(model_path)
        else:
          print "Fresh model"
          model = get_model()
        X, y = load_inria_person(dataset_path)
        print "Training.."
        tensorflow =  TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
        checkpoint_path="weights.{epoch:02d}-loss{loss:.3f}-acc{acc:.3f}-valloss{val_loss:.3f}-valacc{val_acc:.3f}.hdf5"
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
        model.fit(X, y, nb_epoch=50, batch_size=32, shuffle=True, validation_split=0.2, verbose=1, callbacks=[checkpoint, tensorflow])
        predictions = model.predict(np.array([X[0]]))
        print "predictions ", predictions

def launch_quiver(model_path):
  model = load_model(model_path)
  server.launch(model, port=8015,input_folder='./imgs',  classes=['pos', 'neg'])

def visualize_class_activation_map(model_path, paths, output_path):
  with K.tf.device('/gpu'):
        K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)))
        model = load_model(model_path)

        img_paths = []
        if(os.path.isdir(paths)):
          img_paths =  [os.path.join(x[0],y) for x in os.walk(paths) for y in x[2]]
        else:
          img_paths.append(paths)

        for path in img_paths:
          print "Processing image ", path
          original_img = cv2.imread(path, 1)
          width, height, _ = original_img.shape

          #Reshape to the network input shape (3, w, h).
          img = np.array([np.transpose(np.float32(original_img), (2, 0, 1))])
          
          #Get the 512 input weights to the softmax.
          class_weights = model.layers[-1].get_weights()[0]
          final_conv_layer = get_output_layer(model, "conv5_3")
          get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
          [conv_outputs, predictions] = get_output([img])
          # might have to add [img, 1 ] for output as in test mode[conv_outputs, predictions] = get_output([img])
          conv_outputs = conv_outputs[0, :, :, :]

          #Create the class activation map.
          cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[1:3])
          for i, w in enumerate(class_weights[:, 1]):
                  cam += w * conv_outputs[i, :, :]
          print "predictions for image ", path, " is ", predictions
          print "np.count_nonzero(np.isnan(data)) ",  np.count_nonzero(np.isnan(conv_outputs)), " is ", np.count_nonzero(~np.isnan(conv_outputs))
          np.savetxt('convoutputs.txt', conv_outputs.flatten())
          np.savetxt('classweights.txt', class_weights.flatten())
          cam /= np.max(cam)
          np.savetxt('cam.txt', cam)
          cam = cv2.resize(cam, (height, width))
          np.savetxt('cam2.txt', cam)
          heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_WINTER)
          heatmap[np.where(cam < 0.2)] = 0
          img = heatmap*0.5 + original_img
          outpath = output_path + "/" + path.replace("/","_").replace(".","_") + "_pred_"+ str(predictions[0][0]) + "_" + str(predictions[0][1])+"_heatmap.jpg"
          cv2.imwrite(outpath, img)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiver", type = bool, default = False, help = 'Quiver visualization')
    parser.add_argument("--train", type = bool, default = False, help = 'Train the network or visualize a CAM')
    parser.add_argument("--image_path", type = str, help = "Path of an image to run the network on")
    parser.add_argument("--output_path", type = str, default = "heatmapouts/", help = "Path of an image to run the network on")
    parser.add_argument("--model_path", type = str, help = "Path of the trained model")
    parser.add_argument("--dataset_path", type = str, help = \
        'Path to image dataset. Should have pos/neg folders, like in the inria person dataset. \
        http://pascal.inrialpes.fr/data/human/')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = get_args()
        if args.quiver:
            launch_quiver(args.model_path)
        elif args.train:
                train(args.dataset_path, args.model_path)
        else:
                visualize_class_activation_map(args.model_path, args.image_path, args.output_path)
