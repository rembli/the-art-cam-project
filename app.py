####################################################################################
# LIBRARIES
####################################################################################

import sys 
sys.path.append(".")

# Standard libraries
from time import sleep
import datetime
import base64
import binascii
from io import BytesIO
import threading
import functools

# Quart and Quart web socket library
from quart import Quart, render_template, Response, request, redirect, url_for
from quart import websocket
from quart_cors import cors

# libraries to manage images
from PIL import Image
import cv2
import numpy as np

# libraries for neural style transfer
from models import TransformerNet
from utils import *
import torch
from torch.autograd import Variable
import argparse
import os
import tqdm
from torchvision.utils import save_image
from PIL import Image

# library for mongo
import os, yaml, socket
import json
import pymongo
from flask_pymongo import PyMongo
from pymongo.errors import AutoReconnect, PyMongoError
from bson.objectid import ObjectId
import base64
from bson.json_util import dumps


####################################################################################
# UTILITY FUNCTIONS 
####################################################################################

def base64_to_pil_image(base64_img):
    return Image.open(BytesIO(base64.b64decode(base64_img)))

def pil_image_to_base64(pil_image):
    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue())

def pil_image_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil_image(cv2_image):
    img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

def get_remote_addr (request):
    ip = None
    
    # to cover scenarios w/ load balancer
    if request.headers.getlist("X-Forwarded-For"):
        ip = request.headers.getlist("X-Forwarded-For")[0]
    elif request.headers.getlist("X-Real-IP"):
        ip = request.headers.getlist("X-Real-IP")[0]
    else:
        ip = request.remote_addr        
    return str(ip)

def accept_json (request):
    return request.headers.get("accept") == "application/json"

def print_warning(msg):
  sys.stdout.write('%-15s%s\n' % ('WARN:', msg))


def mongo_retry(func, max_retry=5, delay_secs=2):
  """
  Retry decorator for mongodb operations.
  This decorator function allows for retry attempts against a given database
  operation. The purpose is cover situations where the replica set is in a
  state of transition. In the event that an operation is performed against the
  db while the replica set is re-electing a new primary, this function will
  sleep and retry.
  Args:
    max_retry: Maximum number of times to retry a database operation.
    delay_secs: Seconds to wait in between retry attempts.
  """
  def db_op_wrapper(*args, **kwargs):
    count = 0
    while count < max_retry:
      try:
        return func(*args, **kwargs)
      except AutoReconnect:
        print_warning('Op failed to complete...retrying')
        count += 1
        time.sleep(delay_secs)
      except PyMongoError as err:
        return False, str(err)
    return False, str('Op failed after too many retries')
  return db_op_wrapper

####################################################################################
# LOAD AND COMPILE MODEL
####################################################################################

size = 640,640
transform = style_transform()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ohne gpu ist video streaming zu langsam
device = torch.device("cpu")

# Define model and load model checkpoint
transformer = TransformerNet().to(device)
transformer.load_state_dict(torch.load("models/_active_model.pth",map_location='cpu'))
transformer.eval()


####################################################################################
# CAMERA CLASS (with integrated messaging queue)
####################################################################################

class Camera():

    def __init__(self):
        self.to_process = []
        self.to_output = []
        
        thread = threading.Thread(target=self.keep_processing, args=())
        thread.daemon = True
        thread.start() 

    def enqueue_input(self, img):
        # should be larger for videos
        MAX_QUEUE_LENGTH = 1
        if len(self.to_process) < MAX_QUEUE_LENGTH:
            self.to_process.append(img)

    def keep_processing(self):
        while True:
            self.process_one()
            sleep(0.01)        

    def process_one(self):
        if not self.to_process:
            return

        # input is an ascii string 
        input_str = self.to_process.pop(0)

        # convert it to a cv2 image
        input_img = base64_to_pil_image(input_str)

        # resize it
        input_img.thumbnail(size, Image.ANTIALIAS)
        
        # flip it
        # input_img = input_img.transpose(Image.FLIP_LEFT_RIGHT)

        ####################################################################################
        # DO SOME NICE IMAGE MANIPULATION WITH NEURAL STYLE TRANSFER

        # Prepare input
        image_tensor = Variable(transform(input_img)).to(device)
        image_tensor = image_tensor.unsqueeze(0)

        # Stylize image
        with torch.no_grad():
            stylized_image = deprocess (transformer(image_tensor))

        # Convert to PIL image 
        output_img = Image.fromarray(stylized_image.astype('uint8'), 'RGB')

        ####################################################################################
        output_str = pil_image_to_base64(output_img)

        # convert eh base64 string in ascii to base64 string in _bytes_
        self.to_output.append(binascii.a2b_base64(output_str))

    def get_frame(self):
        while not self.to_output:
            sleep(0.05)
        return self.to_output.pop(0)


####################################################################################
# QUART WITH WEB SOCKET SUPPORT
####################################################################################

app = Quart(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

# CORS

app = cors(app, allow_origin="*")

# Init (server-side) camera

camera = Camera()

# CONNNECT TO MONGO

app.config['MONGO_URI'] = os.getenv('MONGO_URI')
mongo = PyMongo(app, retryWrites = True)
db = mongo.db

# HTML TEMPLATES STATIC

@app.route('/')
async def index():
    return await render_template('index.html')


@app.route('/imprint')
async def imprint():
    return await render_template('imprint.html')


@app.route('/about')
async def about():
    return await render_template('about.html')


@app.route('/cam')
async def cam():
    return await render_template('cam.html')

# HTML TEMPLATES DYNAMIC

@mongo_retry
@app.route('/gallery')
async def gallery():
    projection = {"_id": 1, "created_on": 1, "likes": 1, "ip": 1}
    images_top = db.pics.find(projection=projection).limit(3).sort([("likes", pymongo.DESCENDING)])
    images_new = db.pics.find(projection=projection).limit(500).sort([("created_on", pymongo.DESCENDING)])
    if accept_json (request):
        return dumps(images_new)
    return await render_template('gallery.html', images_top=images_top, images_new=images_new)


@mongo_retry
@app.route('/friends')
async def friends():
    guestbook = db.guestbook.find().limit(7).sort([("created_on", pymongo.DESCENDING)])
    return await render_template('friends.html', guestbook=guestbook)


@mongo_retry
@app.route('/guestbook', methods=['POST'])
async def guestbook():
    current_time = datetime.datetime.now() 
    ip = get_remote_addr(request)

    form = await request.form
    text = form['text']
    post = { 
        "created_on": current_time,
        "ip": ip,
        "text": text                
    }
    db.guestbook.insert_one(post)  

    return redirect(url_for('friends'))

@mongo_retry
@app.route('/guestbook/<post_id>', methods=['DELETE'])
async def guestbook_delete_entry(post_id):
    ip = get_remote_addr(request)
    db.guestbook.delete_one({'_id': ObjectId(post_id), "ip": ip })
    return "DELETED"

# APIS

@mongo_retry
@app.route('/ip')
async def ip():
    ip = get_remote_addr(request)
    db.ip.update_one ({ "ip": ip },{ "$set": { "ip": ip } },upsert=True)
    return ip


@mongo_retry
@app.route('/gallery/<image_id>')
async def gallery_image(image_id):
    pic = db.pics.find_one({'_id': ObjectId(image_id)})
    return Response(binascii.a2b_base64(pic["image"]), mimetype='image/jpeg')


@mongo_retry
@app.route('/gallery/<image_id>', methods=['POST'])
async def gallery_image_like(image_id):
    db.pics.find_one_and_update(
        {"_id" : ObjectId(image_id)},
        {"$inc": {"likes": 1}})
    return "LIKED"


@mongo_retry
@app.route('/gallery/<image_id>', methods=['DELETE'])
async def gallery_image_delete(image_id):
    ip = get_remote_addr(request)
    db.pics.delete_one({'_id': ObjectId(image_id), "ip": ip })
    return "DELETED"

# WEB SOCKET ENDPOINT S

@app.websocket('/ws')
async def ws():
    while True:
        data = await websocket.receive()
        jdata = json.loads(data)
        operation = jdata["operation"]
        payload = jdata["payload"]
        image = payload.split(",")[1]

        if operation == "CONVERT":
            camera.enqueue_input(image)
            await websocket.send(camera.get_frame())    

        elif operation == "SHARE":
            image = base64_to_pil_image(image)
            image.thumbnail(size, Image.ANTIALIAS)
            image = pil_image_to_base64 (image)

            current_time = datetime.datetime.now() 
            ip = jdata["ip"]
            pic = { 
                "created_on": current_time,
                "ip": ip,
                "image": image,                
                "likes": 0
            }
            db.pics.insert_one(pic)  

        else:
            print ("unknown operation")

# RUN APP IN DEV / LOCAL ENVIRONMENT

if __name__ == '__main__':
    app.run (host="0.0.0.0", port="5000")
