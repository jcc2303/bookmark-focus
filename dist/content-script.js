console.log('content-script  ???') // client console
// console.log(window, chrome, chrome?.bookmarks, chrome.runtime)

// content-script
// forward message
// window.addListener --> runtime.onMessage()

function sendDomMessage(listenerId, response ){
  if(listenerId) {
    window.dispatchEvent(new CustomEvent(listenerId, {
      detail: response
    }));    
  } else {
    window.postMessage(response)
  }

}


function handleMessageToForward(request) {
  let {listenerId, name, params} = request.data
  if(!name) return 
  
  chrome.runtime.sendMessage({name, params}, async(response) => sendDomMessage(listenerId, response))
}




function init(window){
    console.log('init', window);
    window.addEventListener('message', handleMessageToForward)
    sendDomMessage('chrome-proxy', 'ready')
}


init(window)
