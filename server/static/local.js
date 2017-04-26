'use strict';

function floatTo16BitPCM(output, offset, input){
  for (var i = 0; i < input.length; i++, offset+=2){
    var s = Math.max(-1, Math.min(1, input[i]));
    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
}

function writeString(view, offset, string){
  for (var i = 0; i < string.length; i++){
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function encodeWAV(samples, sampleRate){
  var buffer = new ArrayBuffer(44 + samples.length * 2);
  var view = new DataView(buffer);

  /* RIFF identifier */
  writeString(view, 0, 'RIFF');
  /* file length */
  view.setUint32(4, 32 + samples.length * 2, true);
  /* RIFF type */
  writeString(view, 8, 'WAVE');
  /* format chunk identifier */
  writeString(view, 12, 'fmt ');
  /* format chunk length */
  view.setUint32(16, 16, true);
  /* sample format (raw) */
  view.setUint16(20, 1, true);
  /* channel count */
  view.setUint16(22, 1, true);
  /* sample rate */
  view.setUint32(24, sampleRate, true);
  /* byte rate (sample rate * block align) */
  view.setUint32(28, sampleRate * 2, true);
  /* block align (channel count * bytes per sample) */
  view.setUint16(32, 2, true);
  /* bits per sample */
  view.setUint16(34, 16, true);
  /* data chunk identifier */
  writeString(view, 36, 'data');
  /* data chunk length */
  view.setUint32(40, samples.length * 2, true);

  floatTo16BitPCM(view, 44, samples);

  return view;
}

function WaveObj(divId, src) {
	var wave = Object.create(WaveSurfer);
	var options = {
		container:document.querySelector(divId),
		waveColor:'violet',
		progressColor:'purple',
		cursorColor:'navy'
	};

	wave.init(options);
	wave.load(src);

	this.wave = wave;
	this.onReady = function() {
		var timeline = Object.create(WaveSurfer.Timeline);
        var blind = Object.create(WaveSurfer.Timeline);
        var onespk = Object.create(WaveSurfer.Timeline);
		timeline.init({
			wavesurfer:wave,
			container:"#wave-timeline"
		});
        blind.init({
            wavesurfer:wave,
            container:"#blind"
        });
        onespk.init({
            wavesurfer:wave,
            container:"#onespk"
        });
	}
	this.onError = function(err) {
		console.error(err);
	}
	this.onFinish = function(err) {
		console.log('Finished');
	}
	this.onZoom = function() {
		var val = Number(this.value);
		var expVal = Math.pow(2,val);
		
		console.log('inside onZoom ' + val + ' exp ' + expVal);

		wave.zoom(Math.pow(2,val));
	}

	wave.on('ready', this.onReady);
	wave.on('error', this.onError);
	wave.on('finish', this.onFinish);

	wave.slider = document.getElementById('zoom');
	//wave.slider.value = wave.params.minPxPerSec;
	//wave.slider.min = wave.slider.value;
	wave.slider.addEventListener('change', this.onZoom, false);

    	if (wave.enableDragSelection) {
        	wave.enableDragSelection({
            		color: 'rgba(0, 255, 0, 0.1)'
        	});
    	}
}

var waveObj;

// Init & load audio file
document.addEventListener('DOMContentLoaded', function () {
	waveObj = new WaveObj('#waveform', 'static/demo/F001HJN_F002VAN_001.wav'); 
});

function DumpObject(obj) {
	for (var p in obj) {
		if (p.type != 'function') {
			var v = obj[p];
			console.log(p + '=' + v);
		}
	}
}

function onFileSelect(evt) {
	console.log('inside onFileSelect');
	var files = evt.target.files;
	var output = [];
	for (var i = 0; i < files.length; i++) {
		var f = files[i];
		waveObj.wave.loadBlob(f);
		
	}
}

function createWavBlob() {
	var buf = waveObj.wave.backend.buffer;
	var data = buf.getChannelData(0);
	var wav = encodeWAV(data,buf.sampleRate); 
	var blob = new Blob([wav], {type:'audio/wav'});
	return blob;	
}

function onUpload() {
	console.log('inside upload');

	var buf = waveObj.wave.backend.buffer;
	var data = buf.getChannelData(0);

	var wav = encodeWAV(data,buf.sampleRate); 

	var xhr = new XMLHttpRequest();
	xhr.open('post', '/upload_wav', true);
	var bar = document.querySelector('progress');
	xhr.upload.onprogress = function(e) {
		bar.value = (e.loaded/e.total) * 100;
		bar.textContent = bar.value;	
	};
	console.log('begin send');
	var blob = new Blob([wav], {name:'file', type:'audio/wav'});
    console.log(data)
	xhr.send(blob);
}

function onDownload() {
	console.log('inside download');

	var bar = document.querySelector('progress');
	var xhr = new XMLHttpRequest();
	xhr.open('get', '/upload.wav', true);
	xhr.responseType = 'blob';
	xhr.onprogress = function(e) {
		bar.value = (e.loaded/e.total) * 100;
	}
	xhr.onreadystatechange = function() {
		if (xhr.readyState == 4) {
			console.log('AJAX OK');
			waveObj.wave.loadBlob(xhr.response);
			//var buf = waveObj.wave.backend.buffer;
			//buf.getChannelData(0).set(xhr.response);
		}
	}
	xhr.send(null);
}

function dumpRegions() {
	var list = waveObj.wave.regions.list;
	Object.keys(list).forEach(function(key) {
			var region = list[key];
			console.log('region id: ' + region.id + ' start ' + region.start + ' end ' + region.end);
		}
	);
	console.log('copy it');
}	
	

function genRegion(){
    
}

function onCut() {
	console.log('inside onCut');

	var buf = waveObj.wave.backend.buffer;
	var data = buf.getChannelData(0);
	var head = buf.sampleRate;

	//var start = ~~(Number(document.getElementById('start').value)*buf.sampleRate);
	//var end = ~~(Number(document.getElementById('end').value)*buf.sampleRate);
	var start,end;
	var list = waveObj.wave.regions.list;
	var flag = false;
	Object.keys(list).forEach(function(key) {
			if (!flag) {
				var region = list[key];
				start = region.start;
				end = region.end; 
				flag = true;
			}
		}
	);

	start = ~~(start * buf.sampleRate)
	end = ~~(end * buf.sampleRate);

	var len1 = start;
	var len2 = data.length - end;

	var newData = new Float32Array(len1 + len2);	
	
	var j = 0;
	for (var i = 0; i < start; i++) {
		newData[j++] = data[i];
	} 
	for (var i = end; i < data.length; i++) {
		newData[j++] = data[i]; 
	}
	
	var ctx = waveObj.wave.backend.ac;
	var newBuf = ctx.createBuffer(1, newData.length, buf.sampleRate);
	newBuf.copyToChannel(newData, 0, 0);
	
	waveObj.wave.loadDecodedBuffer(newBuf);
	waveObj.wave.clearRegions();
}

function onSel() {
    console.log('inside onCut');

    var buf = waveObj.wave.backend.buffer;
    var data = buf.getChannelData(0);
    var head = buf.sampleRate;

    //var start = ~~(Number(document.getElementById('start').value)*buf.sampleRate);
    //var end = ~~(Number(document.getElementById('end').value)*buf.sampleRate);
    var start,end;
    var list = waveObj.wave.regions.list;
    var flag = false;
    console.log(list)
    Object.keys(list).forEach(function(key) {
            if (!flag) {
                var region = list[key];
                start = region.start;
                end = region.end; 
                flag = true;
            }
            var region = list[key];
            console.log(region.start, region.end)
        }
    );

    //console.log(start)
    //console.log(end)
    document.getElementById("start_p").value=start;
    document.getElementById("end_p").value=end;
    //start = ~~(start * buf.sampleRate)
    //end = ~~(end * buf.sampleRate);
}


function onCopy() {
	dumpRegions();
}

function onPaste() {
}

function onClear() {
	waveObj.wave.clearRegions();
}

function hookEvent(id,evt,callback) {
	document.getElementById(id).addEventListener(evt,callback,false);
}

function onSave(e) {
	var blob = createWavBlob();
	var a = document.getElementById('saveanchor');
	a.href = URL.createObjectURL(blob);
	console.log('inside onSave');
}

document.getElementById('wavefile').addEventListener('change', onFileSelect,false);
document.getElementById('uploadbutton').addEventListener('click', onUpload,false);
document.getElementById('downloadbutton').addEventListener('click', onDownload,false);
document.getElementById('clearbutton').addEventListener('click', onClear,false);
document.getElementById('savebutton').addEventListener('click', onSave,false);

hookEvent('cutbutton', 'click', onCut);
hookEvent('selbutton', 'click', onSel);
hookEvent('copybutton', 'click', onCopy);
hookEvent('pastebutton', 'click', onPaste);


hookEvent('generate_region', 'click', genRegion)