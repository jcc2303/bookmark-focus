/* src/routes/bookmarks/ClassifierTfjs.svelte generated by Svelte v3.31.1 */
import {
	SvelteComponent,
	append,
	attr,
	check_outros,
	create_component,
	destroy_component,
	detach,
	element,
	group_outros,
	init,
	insert,
	mount_component,
	outro_and_destroy_block,
	safe_not_equal,
	space,
	transition_in,
	transition_out,
	update_keyed_each
} from "../../../web_modules/svelte/internal.js";

import knnWorker from "../../background/knn-worker.js";
import { onMount, onDestroy, createEventDispatcher } from "../../../web_modules/svelte.js";
import Prediction from "./Prediction.js";

function get_each_context(ctx, list, i) {
	const child_ctx = ctx.slice();
	child_ctx[7] = list[i];
	child_ctx[9] = i;
	return child_ctx;
}

function get_each_context_1(ctx, list, i) {
	const child_ctx = ctx.slice();
	child_ctx[7] = list[i];
	child_ctx[9] = i;
	return child_ctx;
}

// (78:4) {#each unsaved as prediction, i (prediction)}
function create_each_block_1(key_1, ctx) {
	let li;
	let prediction;
	let current;

	prediction = new Prediction({
			props: { prediction: /*prediction*/ ctx[7] }
		});

	return {
		key: key_1,
		first: null,
		c() {
			li = element("li");
			create_component(prediction.$$.fragment);
			this.first = li;
		},
		m(target, anchor) {
			insert(target, li, anchor);
			mount_component(prediction, li, null);
			current = true;
		},
		p(new_ctx, dirty) {
			ctx = new_ctx;
			const prediction_changes = {};
			if (dirty & /*unsaved*/ 2) prediction_changes.prediction = /*prediction*/ ctx[7];
			prediction.$set(prediction_changes);
		},
		i(local) {
			if (current) return;
			transition_in(prediction.$$.fragment, local);
			current = true;
		},
		o(local) {
			transition_out(prediction.$$.fragment, local);
			current = false;
		},
		d(detaching) {
			if (detaching) detach(li);
			destroy_component(prediction);
		}
	};
}

// (85:4) {#each saved as prediction, i (prediction)}
function create_each_block(key_1, ctx) {
	let li;
	let prediction;
	let current;

	prediction = new Prediction({
			props: { prediction: /*prediction*/ ctx[7] }
		});

	return {
		key: key_1,
		first: null,
		c() {
			li = element("li");
			create_component(prediction.$$.fragment);
			this.first = li;
		},
		m(target, anchor) {
			insert(target, li, anchor);
			mount_component(prediction, li, null);
			current = true;
		},
		p(new_ctx, dirty) {
			ctx = new_ctx;
			const prediction_changes = {};
			if (dirty & /*saved*/ 1) prediction_changes.prediction = /*prediction*/ ctx[7];
			prediction.$set(prediction_changes);
		},
		i(local) {
			if (current) return;
			transition_in(prediction.$$.fragment, local);
			current = true;
		},
		o(local) {
			transition_out(prediction.$$.fragment, local);
			current = false;
		},
		d(detaching) {
			if (detaching) detach(li);
			destroy_component(prediction);
		}
	};
}

function create_fragment(ctx) {
	let div;
	let ul0;
	let each_blocks_1 = [];
	let each0_lookup = new Map();
	let t;
	let ul1;
	let each_blocks = [];
	let each1_lookup = new Map();
	let current;
	let each_value_1 = /*unsaved*/ ctx[1];
	const get_key = ctx => /*prediction*/ ctx[7];

	for (let i = 0; i < each_value_1.length; i += 1) {
		let child_ctx = get_each_context_1(ctx, each_value_1, i);
		let key = get_key(child_ctx);
		each0_lookup.set(key, each_blocks_1[i] = create_each_block_1(key, child_ctx));
	}

	let each_value = /*saved*/ ctx[0];
	const get_key_1 = ctx => /*prediction*/ ctx[7];

	for (let i = 0; i < each_value.length; i += 1) {
		let child_ctx = get_each_context(ctx, each_value, i);
		let key = get_key_1(child_ctx);
		each1_lookup.set(key, each_blocks[i] = create_each_block(key, child_ctx));
	}

	return {
		c() {
			div = element("div");
			ul0 = element("ul");

			for (let i = 0; i < each_blocks_1.length; i += 1) {
				each_blocks_1[i].c();
			}

			t = space();
			ul1 = element("ul");

			for (let i = 0; i < each_blocks.length; i += 1) {
				each_blocks[i].c();
			}

			attr(ul0, "class", "bg-red-50");
			attr(ul1, "class", "bg-blue-200");
			attr(div, "class", "w-full");
		},
		m(target, anchor) {
			insert(target, div, anchor);
			append(div, ul0);

			for (let i = 0; i < each_blocks_1.length; i += 1) {
				each_blocks_1[i].m(ul0, null);
			}

			append(div, t);
			append(div, ul1);

			for (let i = 0; i < each_blocks.length; i += 1) {
				each_blocks[i].m(ul1, null);
			}

			current = true;
		},
		p(ctx, [dirty]) {
			if (dirty & /*unsaved*/ 2) {
				each_value_1 = /*unsaved*/ ctx[1];
				group_outros();
				each_blocks_1 = update_keyed_each(each_blocks_1, dirty, get_key, 1, ctx, each_value_1, each0_lookup, ul0, outro_and_destroy_block, create_each_block_1, null, get_each_context_1);
				check_outros();
			}

			if (dirty & /*saved*/ 1) {
				each_value = /*saved*/ ctx[0];
				group_outros();
				each_blocks = update_keyed_each(each_blocks, dirty, get_key_1, 1, ctx, each_value, each1_lookup, ul1, outro_and_destroy_block, create_each_block, null, get_each_context);
				check_outros();
			}
		},
		i(local) {
			if (current) return;

			for (let i = 0; i < each_value_1.length; i += 1) {
				transition_in(each_blocks_1[i]);
			}

			for (let i = 0; i < each_value.length; i += 1) {
				transition_in(each_blocks[i]);
			}

			current = true;
		},
		o(local) {
			for (let i = 0; i < each_blocks_1.length; i += 1) {
				transition_out(each_blocks_1[i]);
			}

			for (let i = 0; i < each_blocks.length; i += 1) {
				transition_out(each_blocks[i]);
			}

			current = false;
		},
		d(detaching) {
			if (detaching) detach(div);

			for (let i = 0; i < each_blocks_1.length; i += 1) {
				each_blocks_1[i].d();
			}

			for (let i = 0; i < each_blocks.length; i += 1) {
				each_blocks[i].d();
			}
		}
	};
}

const _knnWorker = new Worker("dist/background/knn-worker.js", { type: "module" });

function instance($$self, $$props, $$invalidate) {
	const dispatch = createEventDispatcher();
	let total = [];
	let saved = [], unsaved = [];

	const classified = async ({ detail }) => {
		console.log("classified", detail);
	};

	// add examples to classifier
	const predicted = async ({ detail }) => {
		dispatch("predicted", detail);

		$$invalidate(2, total = [
			...detail,
			...total.filter(t => !detail.some(d => d.link.url === t.link.url))
		]);

		total.slice(0, 20);
		console.log("total", total);

		// total = Object.values(total.reduce((unique, v) => (unique[v.link.url] = v) && unique  ,{}))
		console.log(total);
	}; // total = total.reduce((acc, value) => acc.some(i => i.link.url === value.link.url) ? acc : acc.concat(value), []);
	// predictions.push(...worked)

	onMount(async () => {
		await createKnnWorker();
	});

	onDestroy(async () => {
		console.log("onDestroy");
		_knnWorker.terminate();
		window.removeEventListener("worker.classified", classified);
		window.removeEventListener("worker.predicted", predicted);
	});

	async function createKnnWorker() {
		// navigator.serviceWorker.register
		window.addEventListener("worker.classified", classified);

		window.addEventListener("worker.predicted", predicted);
		console.log("_knnWorker dispatchEvent");
	}

	$$self.$$.update = () => {
		if ($$self.$$.dirty & /*total, unsaved, saved*/ 7) {
			$: if (total) {
				$$invalidate(1, unsaved = total.filter(({ link, level }) => !link.parentId));
				$$invalidate(0, saved = total.filter(({ link, level }) => link.parentId));
				console.log(unsaved, saved);
			}
		}
	};

	return [saved, unsaved, total];
}

class ClassifierTfjs extends SvelteComponent {
	constructor(options) {
		super();
		init(this, options, instance, create_fragment, safe_not_equal, {});
	}
}

export default ClassifierTfjs;