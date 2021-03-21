import { writable } from '../web_modules/svelte/store.js';

export const filter = writable(localStorage.getItem('filter') || '');
filter.subscribe(val => localStorage.setItem('filter', val) && console.log(val) );

export const overview = writable( null ) 

export const stats = writable( {} ) 

export const bookmarks = writable( null ) 

// it is a folder
export const focused = writable( JSON.parse(localStorage.getItem('focused') || null )  );
focused.subscribe(val => localStorage.setItem('focused', JSON.stringify(val)));




export const backgrounds = writable( JSON.parse(localStorage.getItem('backgrounds') || '{}' )  );
// backgrounds.subscribe(val => localStorage.setItem('backgrounds', JSON.stringify(val)));
