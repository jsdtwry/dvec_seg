from flask import Flask, request, jsonify
from flask import render_template
from compute_feat import *
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'D-vector segmentation tools.'

@app.route('/dvec_seg')
def hello():
    return render_template('dvec_seg.html')

@app.route('/upload_wav', methods=['POST'])
def uplad_wav():
    print 'in upload'
    print request.files
    start = float(request.form['start'])
    end = float(request.form['end'])
    f = request.files['fileupload']
    f.save('wav/t.wav')
    sample_rate('wav/t.wav', 'wav/t_.wav')
    feat_extract('wav/t_.wav', 'conf', '../nnet')
    seglist1 = dvec_seg('wav/t_.wav')
    seglist2 = dvec_seg_one_model('wav/t_.wav', start, end)
    return jsonify(result=[seglist1, seglist2])

def save_result(result, filename):
    f_out = file(filename, 'w')
    for i in result:
        f_out.write(str(i[0][0])+' '+str(i[0][1])+' '+str(i[1])+'\n')
    f_out.close()
    return

@app.route('/seg_1', methods=['POST'])
def seg_1():
    print 'in upload'
    print request.files
    start = float(request.form['start'])
    end = float(request.form['end'])
    f = request.files['fileupload']
    f.save('wav/t.wav')
    sample_rate('wav/t.wav', 'wav/t_.wav')
    feat_extract('wav/t_.wav', 'conf', '../nnet')
    seglist = dvec_seg('wav/t_.wav')
    save_result(seglist, 'static/t1.txt')
    return jsonify(result=seglist)

@app.route('/seg_2', methods=['POST'])
def seg_2():
    print 'in upload'
    print request.files
    start = float(request.form['start'])
    end = float(request.form['end'])
    f = request.files['fileupload']
    f.save('wav/t.wav')
    sample_rate('wav/t.wav', 'wav/t_.wav')
    feat_extract('wav/t_.wav', 'conf', '../nnet')
    seglist = dvec_seg_one_model('wav/t_.wav', start, end)
    save_result(seglist, 'static/t2.txt')
    return jsonify(result=seglist)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

