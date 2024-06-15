#importing packages
import json 
import redis 
import pickle as pkl
from os import path
from pickle import load
from help_funcs import base64_decode_audio,extract_features

#main function to take, preprocess and analyze data from redis 
def consumer():
    db = redis.Redis()
    #real file path was changed due to confidentiality
    path_to_file='/path/to/files/'
    ss = load(open(path_to_file+'scaler.pkl', 'rb'))
    model = pkl.load(open(path_to_file+'LR_audioclass.pickle', 'rb'))
    stop_flag=True
    while stop_flag:
        queue = db.lrange('queue', 0, -1)
        audioIDs = []
        records = []
        for q in queue:
            #deserialize the object
            q = json.loads(q.decode("utf-8"))
            id = q['id']
            record = q['audio']
            audioIDs.append(id)
            records.append(record)
            base64_decode_audio(path_to_file, id, record)
                 
        if len(audioIDs) > 0:
		  # classify the batch
            for id in audioIDs:
                 file = path_to_file + id + '.wav'
                 location = path(file)
                 if location.exists():
                      features = extract_features(file)
                      X = features.reshape(1, -1)
                      X = ss.transform(X)
                      predictions = model.predict(X)
                      #for each prediction create list element and push it to redis-queue
                      for predict in predictions:
                        output = []
                        result = {'result': predict}
                        output.append(result)
                        db.setex(id, 60, json.dumps(output))   
                        db.ltrim('autocall_queue', len(audioIDs), -1) 
        
        if db.exists('autocall_queue') == 0:
             stop_flag=False
             
                         


if __name__ == "__main__":
    # connect to Redis server
    db = redis.Redis(host='127.0.0.1', port=6379, password='null', decode_responses=True)
    consumer()