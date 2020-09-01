# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 14:25:16 2018

@author: SKT
"""

from flask import Flask, request, Response
from flask_cors import CORS
import base64
import json
import facenet
from scipy import misc
from zeep import Client
import datetime

application = Flask(__name__)
CORS(application)

def get_data(request):

    if request.data:
        data = request.data.decode('utf-8')
        data = json.loads(data)

    elif request.form:
        data = request.form

    return data

@application.route('/recognize', methods=['POST'])
def recognize():

    data = get_data(request)

    b64 = data['image']

    if 'data:image/png;base64,' in b64:
        sub = len('data:image/png;base64,')
        b64 = ''.join(list(b64)[sub:])

    print(b64)

    img = base64.b64decode(b64)

    with open('img.jpg', 'wb') as f:
        f.write(img)

    img = misc.imread('img.jpg')
    pred = facenet.predict(img)

    if pred is None:
        return Response()
    else:
        pred_dict, pred_result, pred_score = pred

    data = {}

    id, name = pred_result.split('_')
    data['name'] = name
    data['id'] = id

    data = json.dumps(data)

    print(data)
    print(pred_score)
    print(pred_dict)

    return Response(response=data, status=200, mimetype="application/json")

@application.route('/mark', methods=['POST'])
def mark():

    data = get_data(request)

    print(data)

    emp_id = int(data['User Id'])
    
	# ip = [data['Ip Adddress']]
    ip = request.environ['REMOTE_ADDR']
    
	# curr_date = [data['Calenderdate']]
    curr_date = datetime.datetime.now()

    print(ip, curr_date)

    url = 'http://10.168.50.69/saleswcf_2249.svc?wsdl'
    client = Client(url)

    data = False

    #check  
    result = client.service.SalesLogin(emp_id, curr_date, 'system', 0, 0, ip)
    
    print(result)
    if result == 'VALID':
        #data = True
        #mark
        result = client.service.AttendanceMarkup(emp_id, curr_date, 'system', ip)
        print(result)
        if result == 'Success':
             data = True

    data = json.dumps(data)

    return Response(response=data, status=200)

@application.route('/test')
def test():
    return 'Hello!'

if __name__ == '__main__':
   application.run(host="0.0.0.0", debug = True)
   #application.run()
