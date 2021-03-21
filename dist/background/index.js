console.log('backgroun/index.js trying to inject executeScript');


let chromeApi

async function getChromeApi() {
  const src = chrome.extension.getURL('dist/background/chrome-api.js');
  return (await import(src)).default;
}


async function handleCompletedNavigation (params) {
  console.log('handleCompletedNavigation', params)
  
  let { url, frameId, tabId } = params
  
  if (frameId === 0 && url.startsWith('http://localhost')) {
    chrome.tabs.executeScript(tabId,{ 
      file: 'dist/content-script.js',
      // code: `console.log('injected data');`
    }, (result) => console.log('executed', result))
  } else {
    console.log('no content-script.js inyected on:', url, frameId)
  }
}

function handleMessageToCommand(request, sender, sendResponse) {
  let {name, params} = request
  let method = chromeApi.bookmarks[name] || chromeApi.tabs[name] || chromeApi.extras[name]

  if(!method) {
    console.error(name, params, chromeApi);    
    return false
  }
  method.apply( this, params )
  .then(result => { sendResponse(result) & console.log(name, params, result)  })
  .catch(error => { sendResponse(error) & console.error(name, params, error) });  	

  return true
}


async function start () {
    console.log('Recorder.start', chrome)    
    chrome.webNavigation.onCompleted.addListener(handleCompletedNavigation)
    chromeApi = await getChromeApi()
		chrome.runtime.onMessage.addListener(handleMessageToCommand)
};



start()
