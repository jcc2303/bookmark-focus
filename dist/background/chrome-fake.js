

let data = [{id:0, children: []}];

let last = 0;

export const bookmarks = {

    search (term, callback) {
        let result = this._searchTitle(data[0], term)
        callback( result )
    },
    _searchTitle(node, term ){
        if(node.title === term ) return node
        let found = node.children && node.children.find(x => this._searchTitle(x, term ) )
        return found || []
    },
    setTree (extensionData){
        data = extensionData
        last= 10000
    },
    
    getTree (callback){
        callback(data)
    },

    create (node, callback){
        node.id = last+1
        node.children = []
        let parentNode = this._searchId(data[0], node.parentId)
        if(!parentNode) throw new Error('No parent with id', node.parentId)
        parentNode.children = parentNode.children || []
        parentNode.children = [... parentNode.children, node] 
        callback(node)     
    }, 

    getChildren (id, callback){
        let result = this._searchId(data[0], id)
        callback( result && result.children )
    },
    _searchId(node, id ){
        // console.log('_searchId', node.id, id);
        
        if(node.id == id ) return node
        let result = node.children && node.children.find(x => x.id == id )
        if(result) return result
        else node.children && node.children.find(x => {
            let temp = this._searchId(x, id ) 
            return !!(result = temp)
        })
        return result
    },

    move(id, { parentId }, callback) {
        let child = this._searchId(data[0], id)
        let parent = this._searchId(data[0], parentId)
        parent.children.push(child)
        // remove from old parent
        callback()
    },

    update(id, updated, callback) {
        let result = this._searchId(data[0], id)
        result.title = updated.title
        callback(result)
    },

    get(id, callback) {
        let result = this._searchId(data[0], id)
        // console.log(result);
        callback([result])
    },

    remove(id, callback) {
        callback()
    },

    
}

export const tabs = {
    getSelected (callback) {
        let result = bookmarks._searchId(data[0], "10000")
        callback(result)
    },

    create(node){
        let url = node.url.replace('chrome://newtab', 'http://localhost:8080/override') 
        let win = window.open( url, '_blank'); // about:blank http://localhost:8080/override chrome://newtab
        win && win.focus();    
    }
}

export default {bookmarks, tabs}