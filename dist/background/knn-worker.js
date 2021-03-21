// self.window = self; // egregious hack to get magic-string to work in a worker
console.log('knn-worker');

// import '@tensorflow/tfjs-backend-cpu'
// import '@tensorflow/tfjs-backend-webgl'

import * as tf from '../../web_modules/@tensorflow/tfjs-core.js';

import * as knnClassifier from '../../web_modules/@tensorflow-models/knn-classifier.js';


let classifier

async function classify({detail}) {
    console.log('classify', detail);

    workQueue.push(...detail );
    doStuff()
}

self.addEventListener('worker.classify', classify); 


function loadClassifier() {
    console.log('loadClassifier');
    if (!classifier){
        classifier = knnClassifier.create();
    }
    console.log('classifier', classifier);

}
loadClassifier()


var workQueue = [];

async function doStuff(){
    let predict = workQueue.filter(t => !t.link.classification )
    if(predict.length) await doPrediction(predict)
    workQueue = workQueue.filter(t => t.link.classification )

    let task = workQueue.splice(0, 10) 
    let train = task.filter(t => t.link.classification )
    if(train.length) await doTraining(train)
  

    if (workQueue.length) setTimeout(async ()=> await doStuff(),500);
}


/* {link:{classification}, embedding} =>  undefined */
async function doTraining(task){
    console.log(task.length, workQueue.length);

    await train(task)

    self.dispatchEvent(new CustomEvent('worker.classified', {
        detail: task.length
    }));       
    console.log('classified');
}

/* {link, embedding} =>  { confidences, link } */
async function doPrediction(task){
    console.log(task.length, workQueue.length);

    let result = await predict(task)

    console.log(result);
    self.dispatchEvent(new CustomEvent('worker.predicted', {
        detail: result
    }));       
    console.log('predicted');


    return result
}




const train = async (examples) => {
    console.log('adding samples', examples.length);
    let result = await Promise.all(examples.map(({link, embedding}, i) => classifier.addExample(embedding, link.classification )) )
    console.log('samples added');

    return result;
}

// add examples to classifier
const predict = async (data) => {
    console.log('start predict');
    // test(classifier, encoder, adaptedData)
    let movings = data.map(async({link, embedding}) => {
        let prediction = await classifier.predictClass(embedding);
        let confidences = Object.entries(prediction.confidences).filter(([a, b]) => b > 0 ).sort(([a,b],[c,d])=>d-b) // (${b})
        return { confidences, link }
    })
    console.log('end predict');  

    let predictions = await Promise.all(movings)

    return predictions
}




export default {predict}