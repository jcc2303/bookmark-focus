
// import {bookmarks, tabs} from './chrome-fake'
import {bookmarks, tabs, extras } from './chrome-proxy.js'


let cache = {}


function updateCache(tree, level=0){
    if(level===0) {
        console.log('updateCache',tree)
        cache = {}; 
    }
    cache[tree.id] = tree
    tree.children && tree.children.map(n => updateCache(n, ++level) )
}



const chromeApi = {
	proxied: false,

	bookmarks: {
		setTree: (extensionData) => {
			chrome.bookmarks.setTree(extensionData);
			console.log('faked also data', extensionData);
		},
		
		getTree: () => {		
			return new Promise(resolve => chrome.bookmarks.getTree( result => (result && result.length && updateCache(result[0])) & resolve(result) ) )
		},
	  
		search: (query) => {
			if(typeof query === 'string'){
	
			}
			if(typeof query === 'object'){
					
			}
	
			return new Promise( resolve => chrome.bookmarks.search(query, resolve ) )
		},
		getRecent: (max) => {
			return new Promise( resolve => chrome.bookmarks.getRecent(max, resolve ) )
		},
	
		create: (node) => {
			return new Promise( resolve => chrome.bookmarks.create(node, resolve ))
		},	
	
		update: (id, changes) => {
			return new Promise( resolve => chrome.bookmarks.update(`${id}`, changes, resolve) )
		},
	
		remove: (id) => {
			return new Promise( resolve => chrome.bookmarks.remove(id, resolve))
		},
	
		getChildren: (id) => {
			return new Promise( resolve => chrome.bookmarks.getChildren(id, resolve ) )
		},
	
		get: (id) => {
			let cachedNode = cache[id]
			if(cachedNode) return Promise.resolve ([cachedNode])
			return new Promise( resolve => chrome.bookmarks.get(`${id}`, resolve ))
		},
	
		move: (id, { parentId }) => {
			if (id === parentId) return
			return new Promise( resolve => chrome.bookmarks.move(id, { parentId }, resolve ) )
		},

	},



	// tabs
	tabs: {
		getSelected: () => {
			return new Promise( resolve => chrome.tabs.getSelected(resolve ))
		},
		
		createTab: (url) => {
			return new Promise( resolve => chrome.tabs.create({url}, resolve))
		},

		query: (options) => {
			return new Promise( resolve => chrome.tabs.query(options, resolve ))
		},
	
	},

	// extras
	
	extras: {
		getPath: async (id) => {
			if (!id || /^(0|1|2)$/.test(id) ) return ''
			let result = await chromeApi.bookmarks.get(`${id}`)
			if(!result || !result.length) return ''
			let [folder] = result 
			let path = await chromeApi.extras.getPath(folder.parentId)  
			return  path + '/' + folder.title			
		},
	
		moveChildren: async(children, {parentId}) => {
			let pMoved = children.map( async ({id}) => await chromeApi.bookmarks.move(id, {parentId}) )
			return (await Promise.all(pMoved))
		},		

		
		swapChildren: async (source, destination) => {
			source.children =  await chromeApi.bookmarks.getChildren(source.id)
			destination.children =  await chromeApi.bookmarks.getChildren(destination.id)

			function sortFolderFirst(folders){ return [...folders.filter(x => !x.url), ...folders.filter(x => !!x.url)]}
			await chromeApi.extras.moveChildren(sortFolderFirst(destination.children), {parentId: source.id });

			await chromeApi.extras.moveChildren(sortFolderFirst(source.children), {parentId: destination.id });
		},	
		ping: async(retries)=> {
			return 'bookmark-focus'
		},

		fetchFavicon: (url='') => {
			url = url.replace('https:', 'http:')
			return new Promise( resolve => {
				var xhr = new XMLHttpRequest();
				xhr.open('get','chrome://favicon/'+ url);
				xhr.responseType = 'blob';
				var fr = new FileReader();
				fr.onload = () =>  resolve(fr.result);
			  	xhr.onload = () => fr.readAsDataURL(xhr.response); // async call  				
				xhr.send();
			});		
		}
		  
	},

	operations: {
		getAllLeaf: (tree, level=0) => {
			let {children} = tree
			return [].concat.apply( [...children.filter(n => !!n.url)], children.filter(n => !!n.children).map(n => chromeApi.operations.getAllLeaf(n, ++level)) )
		}
	}
	
}


function checkChrome(chromeX = {}){	
    if(chromeX.bookmarks) {
		console.log('access to chrome bookmark');
	} else {
		chromeApi.proxied = true
		console.log('proxied chrome bookmark and tabs', chromeApi.proxied);
		chromeApi.bookmarks = Object.assign({...chromeApi.bookmarks}, {...bookmarks}) 
		chromeApi.tabs = Object.assign({...chromeApi.tabs}, {...tabs}) 
		chromeApi.extras = Object.assign({...chromeApi.extras}, {...extras}) 
	}
}

checkChrome(chrome)



export default chromeApi
