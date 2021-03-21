/* src/routes/Forrest.svelte generated by Svelte v3.31.1 */
import {
	SvelteComponent,
	add_flush_callback,
	append,
	attr,
	bind,
	binding_callbacks,
	check_outros,
	component_subscribe,
	create_component,
	destroy_component,
	destroy_each,
	detach,
	element,
	group_outros,
	init,
	insert,
	mount_component,
	noop,
	safe_not_equal,
	space,
	subscribe,
	transition_in,
	transition_out
} from "../../web_modules/svelte/internal.js";

import { filter, focused } from "../stores.js";
import { onMount } from "../../web_modules/svelte.js";
import InlineInput from "../../web_modules/svelte-inline-input.js";
import { parse } from "../../web_modules/qs.js";
import { location, querystring } from "../../web_modules/svelte-spa-router.js";
import { overview, bookmarks } from "../stores.js";
import BookmarkFile from "./BookmarkFile.js";
import Analisys from "./analisys/Analisys.js";
import Tailwind from "./Tailwind.js";
import TreeView from "./folders/TreeView.js";
import Overview from "./bookmarks/Overview.js";
import FolderFilter from "./folders/FolderFilter.js";

function get_each_context(ctx, list, i) {
	const child_ctx = ctx.slice();
	child_ctx[6] = list[i];
	child_ctx[7] = list;
	child_ctx[8] = i;
	return child_ctx;
}

// (57:4) {:else}
function create_else_block(ctx) {
	let bookmarkfile;
	let current;
	bookmarkfile = new BookmarkFile({});

	return {
		c() {
			create_component(bookmarkfile.$$.fragment);
		},
		m(target, anchor) {
			mount_component(bookmarkfile, target, anchor);
			current = true;
		},
		i(local) {
			if (current) return;
			transition_in(bookmarkfile.$$.fragment, local);
			current = true;
		},
		o(local) {
			transition_out(bookmarkfile.$$.fragment, local);
			current = false;
		},
		d(detaching) {
			destroy_component(bookmarkfile, detaching);
		}
	};
}

// (52:4) {#each forrest as tree}
function create_each_block(ctx) {
	let treeview;
	let updating_tree;
	let current;

	function treeview_tree_binding(value) {
		/*treeview_tree_binding*/ ctx[5].call(null, value, /*tree*/ ctx[6], /*each_value*/ ctx[7], /*tree_index*/ ctx[8]);
	}

	let treeview_props = {};

	if (/*tree*/ ctx[6] !== void 0) {
		treeview_props.tree = /*tree*/ ctx[6];
	}

	treeview = new TreeView({ props: treeview_props });
	binding_callbacks.push(() => bind(treeview, "tree", treeview_tree_binding));
	treeview.$on("dirty", /*handleDirty*/ ctx[2]);

	return {
		c() {
			create_component(treeview.$$.fragment);
		},
		m(target, anchor) {
			mount_component(treeview, target, anchor);
			current = true;
		},
		p(new_ctx, dirty) {
			ctx = new_ctx;
			const treeview_changes = {};

			if (!updating_tree && dirty & /*forrest*/ 1) {
				updating_tree = true;
				treeview_changes.tree = /*tree*/ ctx[6];
				add_flush_callback(() => updating_tree = false);
			}

			treeview.$set(treeview_changes);
		},
		i(local) {
			if (current) return;
			transition_in(treeview.$$.fragment, local);
			current = true;
		},
		o(local) {
			transition_out(treeview.$$.fragment, local);
			current = false;
		},
		d(detaching) {
			destroy_component(treeview, detaching);
		}
	};
}

function create_fragment(ctx) {
	let div3;
	let div0;
	let t0;
	let div1;
	let overview_1;
	let t1;
	let div2;
	let analisys;
	let current;
	let each_value = /*forrest*/ ctx[0];
	let each_blocks = [];

	for (let i = 0; i < each_value.length; i += 1) {
		each_blocks[i] = create_each_block(get_each_context(ctx, each_value, i));
	}

	const out = i => transition_out(each_blocks[i], 1, 1, () => {
		each_blocks[i] = null;
	});

	let each_1_else = null;

	if (!each_value.length) {
		each_1_else = create_else_block(ctx);
	}

	overview_1 = new Overview({ props: { stats: /*stats*/ ctx[1] } });
	analisys = new Analisys({ props: { stats: /*stats*/ ctx[1] } });

	return {
		c() {
			div3 = element("div");
			div0 = element("div");

			for (let i = 0; i < each_blocks.length; i += 1) {
				each_blocks[i].c();
			}

			if (each_1_else) {
				each_1_else.c();
			}

			t0 = space();
			div1 = element("div");
			create_component(overview_1.$$.fragment);
			t1 = space();
			div2 = element("div");
			create_component(analisys.$$.fragment);
			attr(div0, "class", "px-1 w-1/6 text-align-center antialiased text-gray-900");
			attr(div1, "class", "bg-white w-3/6");
			attr(div2, "class", "bg-gray-100 w-2/6");
			attr(div3, "class", "w-full h-screen bg-gray-100 flex");
		},
		m(target, anchor) {
			insert(target, div3, anchor);
			append(div3, div0);

			for (let i = 0; i < each_blocks.length; i += 1) {
				each_blocks[i].m(div0, null);
			}

			if (each_1_else) {
				each_1_else.m(div0, null);
			}

			append(div3, t0);
			append(div3, div1);
			mount_component(overview_1, div1, null);
			append(div3, t1);
			append(div3, div2);
			mount_component(analisys, div2, null);
			current = true;
		},
		p(ctx, [dirty]) {
			if (dirty & /*forrest, handleDirty*/ 5) {
				each_value = /*forrest*/ ctx[0];
				let i;

				for (i = 0; i < each_value.length; i += 1) {
					const child_ctx = get_each_context(ctx, each_value, i);

					if (each_blocks[i]) {
						each_blocks[i].p(child_ctx, dirty);
						transition_in(each_blocks[i], 1);
					} else {
						each_blocks[i] = create_each_block(child_ctx);
						each_blocks[i].c();
						transition_in(each_blocks[i], 1);
						each_blocks[i].m(div0, null);
					}
				}

				group_outros();

				for (i = each_value.length; i < each_blocks.length; i += 1) {
					out(i);
				}

				check_outros();

				if (each_value.length) {
					if (each_1_else) {
						group_outros();

						transition_out(each_1_else, 1, 1, () => {
							each_1_else = null;
						});

						check_outros();
					}
				} else if (!each_1_else) {
					each_1_else = create_else_block(ctx);
					each_1_else.c();
					transition_in(each_1_else, 1);
					each_1_else.m(div0, null);
				}
			}
		},
		i(local) {
			if (current) return;

			for (let i = 0; i < each_value.length; i += 1) {
				transition_in(each_blocks[i]);
			}

			transition_in(overview_1.$$.fragment, local);
			transition_in(analisys.$$.fragment, local);
			current = true;
		},
		o(local) {
			each_blocks = each_blocks.filter(Boolean);

			for (let i = 0; i < each_blocks.length; i += 1) {
				transition_out(each_blocks[i]);
			}

			transition_out(overview_1.$$.fragment, local);
			transition_out(analisys.$$.fragment, local);
			current = false;
		},
		d(detaching) {
			if (detaching) detach(div3);
			destroy_each(each_blocks, detaching);
			if (each_1_else) each_1_else.d();
			destroy_component(overview_1);
			destroy_component(analisys);
		}
	};
}

let analysis = false;
let showPredictions = false;
let showBookmark = false;

function instance($$self, $$props, $$invalidate) {
	let $bookmarks,
		$$unsubscribe_bookmarks = noop,
		$$subscribe_bookmarks = () => ($$unsubscribe_bookmarks(), $$unsubscribe_bookmarks = subscribe(bookmarks, $$value => $$invalidate(3, $bookmarks = $$value)), bookmarks);

	let $focused;
	component_subscribe($$self, bookmarks, $$value => $$invalidate(3, $bookmarks = $$value));
	component_subscribe($$self, focused, $$value => $$invalidate(4, $focused = $$value));
	$$self.$$.on_destroy.push(() => $$unsubscribe_bookmarks());
	let stats;
	let forrest = [];

	async function handleDirty(event) {
		bookmarks = await dumpBookmarks();
	}

	function treeview_tree_binding(value, tree, each_value, tree_index) {
		each_value[tree_index] = value;
		(($$invalidate(0, forrest), $$invalidate(3, $bookmarks)), $$invalidate(4, $focused));
	}

	$$self.$$.update = () => {
		if ($$self.$$.dirty & /*$bookmarks, $focused*/ 24) {
			$: if ($bookmarks) {
				console.log("$bookmarks", $bookmarks);

				// folders of bookmarkBar and OtherBookmark
				$$invalidate(0, forrest = [
					...!$focused
					? $bookmarks.children[0].children.filter(c => !c.url)
					: [],
					...$bookmarks.children[1].children.filter(c => !c.url)
				]);
			}
		}
	};

	return [forrest, stats, handleDirty, $bookmarks, $focused, treeview_tree_binding];
}

class Forrest extends SvelteComponent {
	constructor(options) {
		super();
		init(this, options, instance, create_fragment, safe_not_equal, {});
	}
}

export default Forrest;