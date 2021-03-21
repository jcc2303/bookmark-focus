import __SNOWPACK_ENV__ from '../../snowpack/env.js';
import.meta.env = __SNOWPACK_ENV__;

import App from "./OverrideApp.js";


console.log("it is app override");


var app = new App({
  target: document.body,
});

export default app;

// Hot Module Replacement (HMR) - Remove this snippet to remove HMR.
// Learn more: https://www.snowpack.dev/#hot-module-replacement
if (import.meta.hot) {
  import.meta.hot.accept();
  import.meta.hot.dispose(() => {
    app.$destroy();
  });
}
