

let data = [{id:0, children: []}];

let last = 0;


let iconsCache = {}

let cache = {}


function updateCache(tree, level=0){
    if(level===0) {
        console.log('updateCache',tree)
        cache = {}; 
    }
    cache[tree.id] = tree
    tree.children && tree.children.map(n => updateCache(n, ++level) )
}


export const ERROR = {error: "timeout"}

// (afterSave(arg) or whatever) and your onSave handler can then use event.target.afterSave(result)
const sendMessage = (name, params, callback) => {    
    let listenerId = Math.random().toString(36).substr(2, 9);
    let randomListener = function(event) { // random name
        // console.log(listenerId, event, callback);
        if (event.detail != ERROR) ;// console.log(listenerId, name, params, event.detail)
        else console.warn(listenerId, name, params, event) 
        callback && callback(event.detail)
        window.removeEventListener(listenerId, randomListener)
        randomListener = null
    }
    
    // additional details come with the event to the handler
    window.addEventListener(listenerId, randomListener);    
    window.postMessage({name, params, listenerId})
    setTimeout(() => randomListener && randomListener({detail: ERROR}), 10000 )
    // call api send random name to proxy
    // proxy send event with // random name 

    // , async(response) => console.log('response after getTree', response) 
}



export const bookmarks = {
    setTree (extensionData){
        data = extensionData
        last= 10000
    },

    search (term, callback) {
        return new Promise(async(resolve, reject) => sendMessage( 'search', [term], (r) =>  r !== ERROR ? resolve(r): reject(null)))
    },
    getRecent (term, callback) {
        return new Promise(async(resolve, reject) => sendMessage( 'getRecent', [term], (r) => r !== ERROR ? resolve(r): reject(null)))
    },
    
    getTree (callback){     
        return new Promise(async(resolve, reject) => sendMessage( 'getTree', [], (r) => r !== ERROR ? (r && r.length && updateCache(r[0])) & resolve(r): reject(null)  ))
    },

    create (node, callback){        
        return new Promise(async(resolve, reject) => sendMessage('create', [node], (r) => r !== ERROR ? resolve(r): reject(null)))
    }, 

    getChildren (id, callback){
        return new Promise(async(resolve, reject) => sendMessage('getChildren', [id], (r) => r !== ERROR ? resolve(r): reject(null)))
    },

    move(id, destination, callback) {
        return new Promise(async(resolve, reject) => sendMessage('move', [id, destination], (r) => r !== ERROR ? resolve(r): reject(null)))
    },

    update(id, changes, callback) {
        return new Promise(async(resolve, reject) => sendMessage('update', [id, changes], (r) => r !== ERROR ? resolve(r): reject(null)))
    },

    get(id, callback) {
        let cachedNode = cache[id]
        if(cachedNode) return Promise.resolve ([cachedNode])
        return new Promise(async(resolve, reject) => sendMessage('get', [id], (r) => r !== ERROR ? resolve(r): reject(null)))
    },

    remove(id, callback) {
        return new Promise(async(resolve, reject) => sendMessage('remove', [id], (r) => r !== ERROR ? resolve(r): reject(null)))
    },

    
}

export const tabs = {
    getSelected () {
        return new Promise(async(resolve, reject) => sendMessage('getSelected', [], resolve )).catch(x => console.log(x) )
    },

    createTab(url){
        if( /^http|https|chrome\:\/\/newtab/.test(url) ) { // window.location.href
            url = url.replace('chrome://newtab', '/override') 
            let win = window.open( url, '_blank'); // about:blank http://localhost:8080/override chrome://newtab
            win && win.focus();
            return
        } 
        sendMessage('createTab', [url])
    },

    query (options) {
        return new Promise(async(resolve) => sendMessage('query', [options], resolve )).catch(x => console.log(x) )
    },


}

export const extras = {
    swapChildren: async (source, destination) => {
        source =  {id: source.id }
        destination = { id: destination.id }
        return new Promise(async(resolve, reject) => sendMessage('swapChildren', [source, destination],(r) => r !== ERROR ? resolve(r): reject(null))) // .catch(x => console.log(x) )
    },

    moveChildren: async(children, destination) => {
        return new Promise(async(resolve) => sendMessage('moveChildren', [children, destination, resolve])).catch(x => console.log(x) )
    },

    ping: async (retries=3) => {
        console.log('ping retries', retries);
        if (retries < 1 ) return
        
        let result = await new Promise(async(resolve) =>
            sendMessage('ping', [], (r) => r !== ERROR ? resolve(r): reject(null) )
        )

        if(result) return result
        else return await extras.ping(retries -1)

    },
    fetchFavicon: async (url) => {
        const parsedUrl = new URL(url)
        let cachedIcon = iconsCache[parsedUrl.hostname] // iconsCache[url]
        if(cachedIcon) return Promise.resolve ([cachedIcon])	
        
        return new Promise(async(resolve) => {
            sendMessage('fetchFavicon', [url], resolve )
        }).then(r => {
            r && (iconsCache[parsedUrl.hostname] = r)
            return r
        }).catch(x => console.log(x) )

    }

}

export default {bookmarks, tabs, extras}