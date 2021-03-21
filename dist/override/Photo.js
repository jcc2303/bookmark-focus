
import {stringify} from '../../web_modules/qs.js'


const unsplashAPI = 'https://api.unsplash.com/search/photos'
const collectionsByPeriod = {
	day: [1507483],
	night: [4747434],
	dawn: [4748158],
	dusk: [4748158],
};

export async function fetchUnsplash({period = 'day', query}) {
    let unsplash =  query? unsplashAPI : 'https://api.unsplash.com/photos/random'
	const params = {
        query,
		orientation: 'landscape',
		w: 1920,
		// my collection with curated photos, refer to https://unsplash.com/@trongthanh/collections
		collections: 1507483 // collectionsByPeriod[period] || collectionsByPeriod.day,
	};

	// NOTE: the client id belongs to Nau-Tab only, please request your own Application at Unsplash
	const headers = new Headers({
		'Content-Type': 'application/json',
		Accept: 'application/json',
		Authorization: 'Client-ID svGLnoJYk5XBIsIMrmigQkEu7gRRT62WFpgcB4dRAPI',
	});

	const req = new Request(`${unsplash}?${stringify(params)}`, {
		method: 'GET',
		headers
	});

	let response = await fetch(req).catch(err => {
			console.log('Errors:', err);
			return err;
	});

	return response?.json()
}
