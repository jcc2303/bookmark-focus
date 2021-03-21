/* src/override/OverrideApp.svelte generated by Svelte v3.31.1 */
import {
	SvelteComponent,
	append,
	attr,
	check_outros,
	component_subscribe,
	create_component,
	destroy_component,
	detach,
	element,
	group_outros,
	init,
	insert,
	mount_component,
	safe_not_equal,
	set_store_value,
	space,
	transition_in,
	transition_out
} from "../../web_modules/svelte/internal.js";

import { onMount } from "../../web_modules/svelte.js";
import { CrosshairIcon, ImageIcon } from "../../web_modules/svelte-feather-icons.js";
import Navbar from "./Navbar.js";
import { fetchUnsplash } from "./Photo.js";
import { overview, backgrounds, stats, bookmarks } from "../stores.js";
import Forrest from "../routes/Forrest.js";
import chromeApi from "../background/chrome-api.js";
import statsApi from "../background/stats.js";

function create_if_block(ctx) {
	let forrest;
	let current;
	forrest = new Forrest({});

	return {
		c() {
			create_component(forrest.$$.fragment);
		},
		m(target, anchor) {
			mount_component(forrest, target, anchor);
			current = true;
		},
		i(local) {
			if (current) return;
			transition_in(forrest.$$.fragment, local);
			current = true;
		},
		o(local) {
			transition_out(forrest.$$.fragment, local);
			current = false;
		},
		d(detaching) {
			destroy_component(forrest, detaching);
		}
	};
}

function create_fragment(ctx) {
	let main;
	let navbar;
	let t;
	let current;
	navbar = new Navbar({});
	navbar.$on("refresh", /*dumpBookmarks*/ ctx[1]);
	let if_block = /*overviewId*/ ctx[0] !== undefined && create_if_block(ctx);

	return {
		c() {
			main = element("main");
			create_component(navbar.$$.fragment);
			t = space();
			if (if_block) if_block.c();
			attr(main, "class", "font-sans text-sm");
		},
		m(target, anchor) {
			insert(target, main, anchor);
			mount_component(navbar, main, null);
			append(main, t);
			if (if_block) if_block.m(main, null);
			current = true;
		},
		p(ctx, [dirty]) {
			if (/*overviewId*/ ctx[0] !== undefined) {
				if (if_block) {
					if (dirty & /*overviewId*/ 1) {
						transition_in(if_block, 1);
					}
				} else {
					if_block = create_if_block(ctx);
					if_block.c();
					transition_in(if_block, 1);
					if_block.m(main, null);
				}
			} else if (if_block) {
				group_outros();

				transition_out(if_block, 1, 1, () => {
					if_block = null;
				});

				check_outros();
			}
		},
		i(local) {
			if (current) return;
			transition_in(navbar.$$.fragment, local);
			transition_in(if_block);
			current = true;
		},
		o(local) {
			transition_out(navbar.$$.fragment, local);
			transition_out(if_block);
			current = false;
		},
		d(detaching) {
			if (detaching) detach(main);
			destroy_component(navbar);
			if (if_block) if_block.d();
		}
	};
}

function getParam(param) {
	let params = new URL(document.location).searchParams;
	return params.get(param);
}

function instance($$self, $$props, $$invalidate) {
	let $bookmarks;
	let $overview;
	let $stats;
	component_subscribe($$self, bookmarks, $$value => $$invalidate(2, $bookmarks = $$value));
	component_subscribe($$self, overview, $$value => $$invalidate(3, $overview = $$value));
	component_subscribe($$self, stats, $$value => $$invalidate(4, $stats = $$value));
	let tree = { id: "1", children: [] };
	let _children = [], _links = [];

	// folder.subscribe(val => localStorage.setItem('folder', JSON.stringify(val)));
	let overviewId;

	onMount(async () => {
		$$invalidate(0, overviewId = getParam("folderId") || 0);
		console.log("overviewId", overviewId);

		// await checkBackground($folder)    
		set_store_value(bookmarks, $bookmarks = await dumpBookmarks(), $bookmarks);

		console.log("bookmarks", $bookmarks);

		set_store_value(
			overview,
			$overview = overviewId
			? (await chromeApi.bookmarks.get(`${overviewId}`))[0]
			: $bookmarks,
			$overview
		);
	});

	const dumpBookmarks = async () => {
		console.log("dumpBookmarks");
		let currentTab = await chromeApi.tabs.getSelected();
		let bookmarkTreeNodes = await chromeApi.bookmarks.getTree();
		let bookmarks = bookmarkTreeNodes && bookmarkTreeNodes[0];
		set_store_value(stats, $stats = statsApi.generateStats(bookmarks), $stats);
		return bookmarks;
	};

	return [overviewId, dumpBookmarks];
}

class OverrideApp extends SvelteComponent {
	constructor(options) {
		super();
		init(this, options, instance, create_fragment, safe_not_equal, {});
	}
}

export default OverrideApp;