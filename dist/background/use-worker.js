// self.window = self; // egregious hack to get magic-string to work in a worker
console.log('use-worker');


import '../../web_modules/@tensorflow/tfjs-backend-cpu.js'
import '../../web_modules/@tensorflow/tfjs-backend-webgl.js'

import * as use from '../../web_modules/@tensorflow-models/universal-sentence-encoder.js'

// async function getChromeApi() {
//     const src = chrome.extension.getURL('dist/background/chrome-api.js');
//     return (await import(src)).default;
// }


let model
let pmodel

async function loadUseModel() {
    console.log('loadUseModel');
    if (!model){
        pmodel = use.load().then(m => model = m)
    }
}
loadUseModel()


var workQueue = [];

async function doStuff(){
  if (!databaseBusy()) await doStuffIndeed(workQueue.splice(0, 10)); // .shift()
  if (workQueue.length) setTimeout(async ()=> await doStuff(),500);
}

function databaseBusy(){
    return false
}

async function doStuffIndeed(task){
    console.log(task.length, workQueue.length);
    let examples = await embed(task)

    self.dispatchEvent(new CustomEvent('worker.embedded', {
        detail: examples
    }));       
 
    return examples   
}

async function analisys({detail}) {
    console.log('analisys', detail);
    if(!model) await pmodel

    if(detail[0].classification) workQueue.unshift(...detail);
    else workQueue.push(...detail);

    await doStuff()
}
self.addEventListener('worker.analisys', analisys); 




async function embed (data){
    console.log('embed', data.length);
    let examples = await Promise.all(data.map(async (link) => {
        let embedding = await model.embed(link.title)
        return {link, embedding} 
    }))
    return examples
}


export default {embed}