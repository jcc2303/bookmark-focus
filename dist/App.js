/* src/App.svelte generated by Svelte v3.31.1 */
import {
	SvelteComponent,
	append,
	attr,
	check_outros,
	create_component,
	destroy_component,
	detach,
	element,
	empty,
	group_outros,
	init,
	insert,
	mount_component,
	noop,
	safe_not_equal,
	set_data,
	text,
	transition_in,
	transition_out
} from "../web_modules/svelte/internal.js";

import { onMount } from "../web_modules/svelte.js";
import Router from "../web_modules/svelte-spa-router.js";
import AddBookmark from "./routes/bookmarks/AddBookmark.js";
import Folders from "./routes/folders/Folders.js";
import Bookmarks from "./routes/bookmarks/Bookmarks.js";
import NotFound from "./routes/NotFound.js";
import chromeApi from "./background/chrome-api.js";

function create_else_block(ctx) {
	let p;
	let t0;
	let t1;

	return {
		c() {
			p = element("p");
			t0 = text("extension is active?");
			t1 = text(/*ping*/ ctx[0]);
		},
		m(target, anchor) {
			insert(target, p, anchor);
			append(p, t0);
			append(p, t1);
		},
		p(ctx, dirty) {
			if (dirty & /*ping*/ 1) set_data(t1, /*ping*/ ctx[0]);
		},
		i: noop,
		o: noop,
		d(detaching) {
			if (detaching) detach(p);
		}
	};
}

// (72:0) {#if ping}
function create_if_block(ctx) {
	let main;
	let router;
	let current;
	router = new Router({ props: { routes: /*routes*/ ctx[1] } });

	return {
		c() {
			main = element("main");
			create_component(router.$$.fragment);
			attr(main, "class", "font-sans text-sm");
		},
		m(target, anchor) {
			insert(target, main, anchor);
			mount_component(router, main, null);
			current = true;
		},
		p: noop,
		i(local) {
			if (current) return;
			transition_in(router.$$.fragment, local);
			current = true;
		},
		o(local) {
			transition_out(router.$$.fragment, local);
			current = false;
		},
		d(detaching) {
			if (detaching) detach(main);
			destroy_component(router);
		}
	};
}

function create_fragment(ctx) {
	let current_block_type_index;
	let if_block;
	let if_block_anchor;
	let current;
	const if_block_creators = [create_if_block, create_else_block];
	const if_blocks = [];

	function select_block_type(ctx, dirty) {
		if (/*ping*/ ctx[0]) return 0;
		return 1;
	}

	current_block_type_index = select_block_type(ctx, -1);
	if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);

	return {
		c() {
			if_block.c();
			if_block_anchor = empty();
		},
		m(target, anchor) {
			if_blocks[current_block_type_index].m(target, anchor);
			insert(target, if_block_anchor, anchor);
			current = true;
		},
		p(ctx, [dirty]) {
			let previous_block_index = current_block_type_index;
			current_block_type_index = select_block_type(ctx, dirty);

			if (current_block_type_index === previous_block_index) {
				if_blocks[current_block_type_index].p(ctx, dirty);
			} else {
				group_outros();

				transition_out(if_blocks[previous_block_index], 1, 1, () => {
					if_blocks[previous_block_index] = null;
				});

				check_outros();
				if_block = if_blocks[current_block_type_index];

				if (!if_block) {
					if_block = if_blocks[current_block_type_index] = if_block_creators[current_block_type_index](ctx);
					if_block.c();
				} else {
					if_block.p(ctx, dirty);
				}

				transition_in(if_block, 1);
				if_block.m(if_block_anchor.parentNode, if_block_anchor);
			}
		},
		i(local) {
			if (current) return;
			transition_in(if_block);
			current = true;
		},
		o(local) {
			transition_out(if_block);
			current = false;
		},
		d(detaching) {
			if_blocks[current_block_type_index].d(detaching);
			if (detaching) detach(if_block_anchor);
		}
	};
}

function routeLoading(event) {
	console.log("routeLoading event");
	console.log("Route", event.detail.route);
	console.log("Location", event.detail.location);
	console.log("Querystring", event.detail.querystring);
	console.log("User data", event.detail.userData);
}

function routeLoaded(event) {
	console.log("routeLoaded event");

	// The first 4 properties are the same as for the routeLoading event
	console.log("Route", event.detail.route);

	console.log("Location", event.detail.location);
	console.log("Querystring", event.detail.querystring);
	console.log("User data", event.detail.userData);

	// The last two properties are unique to routeLoaded
	console.log("Component", event.detail.component); // This is a Svelte component, so a function

	console.log("Name", event.detail.name);
}

function instance($$self, $$props, $$invalidate) {
	const routes = {
		// Exact path
		"/": AddBookmark,
		// Wildcard parameter
		"/bookmarks/*": Bookmarks,
		"/folders/*": Folders,
		// Catch-all
		"*": NotFound
	};

	let ping;
	const ERROR = {};

	function waitForChromeProxy() {
		console.log("waitForChromeProxy");
		let listenerId = "chrome-proxy";

		const proxyChromeListener = event => {
			console.log("timeout?", event);
			if (event.detail === ERROR) window.location.href = window.location.href + "/override";
			console.log(event);
			$$invalidate(0, ping = {});
			console.log("listener-ping", ping);
			window.removeEventListener(listenerId, proxyChromeListener);
		};

		window.addEventListener(listenerId, proxyChromeListener);
		setTimeout(() => proxyChromeListener({ detail: ERROR }), 10000);
	}

	onMount(() => {
		if (chromeApi.proxied) waitForChromeProxy(); else $$invalidate(0, ping = true);
	}); // ping = await chromeApi.extras.ping()
	// console.log('ping', ping);

	return [ping, routes];
}

class App extends SvelteComponent {
	constructor(options) {
		super();
		init(this, options, instance, create_fragment, safe_not_equal, {});
	}
}

export default App;