// import nlp from 'compromise'

// import keyword_extractor from 'keyword-extractor'

import keyword from './en.js'
console.log(keyword);

let cache = {}

let stats = {}

let tags

let level
function updateCache(tree, level=0, path='', parent){
    if(level===0) {
        console.log('updateCache',tree)
        cache = {}; 
    }
    let {children, tags, title} = tree
    cache[tree.id] = tree
    tree.level = level++ 
    tree.path = path = path + '/' +title
    tree.parent = parent
    if(children){
        if(!tags){
            let tags = extractTags(tree)
            tree.tags = tags
            children.map(n => {
                updateCache(n, level, path, tree ) 
            })   
        }
    }
}

function extractTags({children}) {
    let regex = /[!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~·•—–]/g;
    let sentencesWordsList  = children.filter(node => node.url).flat().map(({title}) => title.toLowerCase() )
    let splitSentences = sentencesWordsList.map(title => title.replace(regex, ' ').split(/\s/).filter(x => x) ).flat()
    let wordsByLinks = removeStopWord( splitSentences )
    
    let wc =  wordsCount(wordsByLinks)
    let sortedWords = Object.entries(wc).sort(([a,b],[c,d])=>d-b)
    let result = sortedWords// .map(([index, value]) => index) // //.slice(0,reduced)
    return result
}

function wordsCount(wordsList){
    let result = {}
    wordsList.map(w => result[w] = (result[w] || 0 )+1 ) 
    return result
}

function removeStopWord(words){
    let {stopwords} = keyword
    return words.filter(w => !stopwords.includes(w))
}

function generateStats(folder){
    updateCache(folder, level=0)
    stats.cache = cache
    // \w[.?!](\s|$)
    // number of sentences contained in the text,
    // number of words in each sentence,
    // number of letters in each word,
    // average number of words per sentence, and
    // average word length.

    return stats
} 


const statsApi = {
    generateStats, stats
}

export default statsApi 