from flask import Flask, request, jsonify
from flask import render_template
from compute_feat import *
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'D-vector segmentation tools.'

@app.route('/dvec_seg')
def hello():
    return render_template('hello.html')

@app.route('/upload_wav', methods=['POST'])
def uplad_wav():
    print 'in upload'
    print request.files
    start = float(request.form['start'])
    end = float(request.form['end'])
    f = request.files['fileupload']
    f.save('wav/t.wav')
    feat_extract('wav/t.wav', 'conf', '../nnet')
    seglist1 = dvec_seg('wav/t.wav')
    seglist2 = dvec_seg_one_model('wav/t.wav', start, end)
    return jsonify(result=[seglist1, seglist2])

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

