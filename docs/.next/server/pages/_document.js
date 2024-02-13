"use strict";
/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
(() => {
var exports = {};
exports.id = "pages/_document";
exports.ids = ["pages/_document"];
exports.modules = {

/***/ "./src/pages/_document.jsx":
/*!*********************************!*\
  !*** ./src/pages/_document.jsx ***!
  \*********************************/
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export */ __webpack_require__.d(__webpack_exports__, {\n/* harmony export */   \"default\": () => (/* binding */ Document)\n/* harmony export */ });\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! react/jsx-dev-runtime */ \"react/jsx-dev-runtime\");\n/* harmony import */ var react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__);\n/* harmony import */ var next_document__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! next/document */ \"./node_modules/next/document.js\");\n/* harmony import */ var next_document__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(next_document__WEBPACK_IMPORTED_MODULE_1__);\n\n\nconst modeScript = `\n  let darkModeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)')\n\n  updateMode()\n  darkModeMediaQuery.addEventListener('change', updateModeWithoutTransitions)\n  window.addEventListener('storage', updateModeWithoutTransitions)\n\n  function updateMode() {\n    let isSystemDarkMode = darkModeMediaQuery.matches\n    let isDarkMode = window.localStorage.isDarkMode === 'true' || (!('isDarkMode' in window.localStorage) && isSystemDarkMode)\n\n    if (isDarkMode) {\n      document.documentElement.classList.add('dark')\n    } else {\n      document.documentElement.classList.remove('dark')\n    }\n\n    if (isDarkMode === isSystemDarkMode) {\n      delete window.localStorage.isDarkMode\n    }\n  }\n\n  function disableTransitionsTemporarily() {\n    document.documentElement.classList.add('[&_*]:!transition-none')\n    window.setTimeout(() => {\n      document.documentElement.classList.remove('[&_*]:!transition-none')\n    }, 0)\n  }\n\n  function updateModeWithoutTransitions() {\n    disableTransitionsTemporarily()\n    updateMode()\n  }\n`;\nfunction Document() {\n    return /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(next_document__WEBPACK_IMPORTED_MODULE_1__.Html, {\n        lang: \"en\",\n        children: [\n            /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(next_document__WEBPACK_IMPORTED_MODULE_1__.Head, {\n                children: /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"script\", {\n                    dangerouslySetInnerHTML: {\n                        __html: modeScript\n                    }\n                }, void 0, false, {\n                    fileName: \"/Users/nicolasalan/microvault/microvault/docs/src/pages/_document.jsx\",\n                    lineNumber: 42,\n                    columnNumber: 9\n                }, this)\n            }, void 0, false, {\n                fileName: \"/Users/nicolasalan/microvault/microvault/docs/src/pages/_document.jsx\",\n                lineNumber: 41,\n                columnNumber: 7\n            }, this),\n            /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(\"body\", {\n                className: \"bg-white antialiased dark:bg-zinc-900\",\n                children: [\n                    /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(next_document__WEBPACK_IMPORTED_MODULE_1__.Main, {}, void 0, false, {\n                        fileName: \"/Users/nicolasalan/microvault/microvault/docs/src/pages/_document.jsx\",\n                        lineNumber: 45,\n                        columnNumber: 9\n                    }, this),\n                    /*#__PURE__*/ (0,react_jsx_dev_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxDEV)(next_document__WEBPACK_IMPORTED_MODULE_1__.NextScript, {}, void 0, false, {\n                        fileName: \"/Users/nicolasalan/microvault/microvault/docs/src/pages/_document.jsx\",\n                        lineNumber: 46,\n                        columnNumber: 9\n                    }, this)\n                ]\n            }, void 0, true, {\n                fileName: \"/Users/nicolasalan/microvault/microvault/docs/src/pages/_document.jsx\",\n                lineNumber: 44,\n                columnNumber: 7\n            }, this)\n        ]\n    }, void 0, true, {\n        fileName: \"/Users/nicolasalan/microvault/microvault/docs/src/pages/_document.jsx\",\n        lineNumber: 40,\n        columnNumber: 5\n    }, this);\n}\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiLi9zcmMvcGFnZXMvX2RvY3VtZW50LmpzeCIsIm1hcHBpbmdzIjoiOzs7Ozs7Ozs7QUFBNEQ7QUFFNUQsTUFBTUksYUFBYSxDQUFDOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFpQ3BCLENBQUM7QUFFYyxTQUFTQztJQUN0QixxQkFDRSw4REFBQ0osK0NBQUlBO1FBQUNLLE1BQUs7OzBCQUNULDhEQUFDTiwrQ0FBSUE7MEJBQ0gsNEVBQUNPO29CQUFPQyx5QkFBeUI7d0JBQUVDLFFBQVFMO29CQUFXOzs7Ozs7Ozs7OzswQkFFeEQsOERBQUNNO2dCQUFLQyxXQUFVOztrQ0FDZCw4REFBQ1QsK0NBQUlBOzs7OztrQ0FDTCw4REFBQ0MscURBQVVBOzs7Ozs7Ozs7Ozs7Ozs7OztBQUluQiIsInNvdXJjZXMiOlsid2VicGFjazovL21pY3JvdmF1bHQvLi9zcmMvcGFnZXMvX2RvY3VtZW50LmpzeD9hMDQxIl0sInNvdXJjZXNDb250ZW50IjpbImltcG9ydCB7IEhlYWQsIEh0bWwsIE1haW4sIE5leHRTY3JpcHQgfSBmcm9tICduZXh0L2RvY3VtZW50J1xuXG5jb25zdCBtb2RlU2NyaXB0ID0gYFxuICBsZXQgZGFya01vZGVNZWRpYVF1ZXJ5ID0gd2luZG93Lm1hdGNoTWVkaWEoJyhwcmVmZXJzLWNvbG9yLXNjaGVtZTogZGFyayknKVxuXG4gIHVwZGF0ZU1vZGUoKVxuICBkYXJrTW9kZU1lZGlhUXVlcnkuYWRkRXZlbnRMaXN0ZW5lcignY2hhbmdlJywgdXBkYXRlTW9kZVdpdGhvdXRUcmFuc2l0aW9ucylcbiAgd2luZG93LmFkZEV2ZW50TGlzdGVuZXIoJ3N0b3JhZ2UnLCB1cGRhdGVNb2RlV2l0aG91dFRyYW5zaXRpb25zKVxuXG4gIGZ1bmN0aW9uIHVwZGF0ZU1vZGUoKSB7XG4gICAgbGV0IGlzU3lzdGVtRGFya01vZGUgPSBkYXJrTW9kZU1lZGlhUXVlcnkubWF0Y2hlc1xuICAgIGxldCBpc0RhcmtNb2RlID0gd2luZG93LmxvY2FsU3RvcmFnZS5pc0RhcmtNb2RlID09PSAndHJ1ZScgfHwgKCEoJ2lzRGFya01vZGUnIGluIHdpbmRvdy5sb2NhbFN0b3JhZ2UpICYmIGlzU3lzdGVtRGFya01vZGUpXG5cbiAgICBpZiAoaXNEYXJrTW9kZSkge1xuICAgICAgZG9jdW1lbnQuZG9jdW1lbnRFbGVtZW50LmNsYXNzTGlzdC5hZGQoJ2RhcmsnKVxuICAgIH0gZWxzZSB7XG4gICAgICBkb2N1bWVudC5kb2N1bWVudEVsZW1lbnQuY2xhc3NMaXN0LnJlbW92ZSgnZGFyaycpXG4gICAgfVxuXG4gICAgaWYgKGlzRGFya01vZGUgPT09IGlzU3lzdGVtRGFya01vZGUpIHtcbiAgICAgIGRlbGV0ZSB3aW5kb3cubG9jYWxTdG9yYWdlLmlzRGFya01vZGVcbiAgICB9XG4gIH1cblxuICBmdW5jdGlvbiBkaXNhYmxlVHJhbnNpdGlvbnNUZW1wb3JhcmlseSgpIHtcbiAgICBkb2N1bWVudC5kb2N1bWVudEVsZW1lbnQuY2xhc3NMaXN0LmFkZCgnWyZfKl06IXRyYW5zaXRpb24tbm9uZScpXG4gICAgd2luZG93LnNldFRpbWVvdXQoKCkgPT4ge1xuICAgICAgZG9jdW1lbnQuZG9jdW1lbnRFbGVtZW50LmNsYXNzTGlzdC5yZW1vdmUoJ1smXypdOiF0cmFuc2l0aW9uLW5vbmUnKVxuICAgIH0sIDApXG4gIH1cblxuICBmdW5jdGlvbiB1cGRhdGVNb2RlV2l0aG91dFRyYW5zaXRpb25zKCkge1xuICAgIGRpc2FibGVUcmFuc2l0aW9uc1RlbXBvcmFyaWx5KClcbiAgICB1cGRhdGVNb2RlKClcbiAgfVxuYFxuXG5leHBvcnQgZGVmYXVsdCBmdW5jdGlvbiBEb2N1bWVudCgpIHtcbiAgcmV0dXJuIChcbiAgICA8SHRtbCBsYW5nPVwiZW5cIj5cbiAgICAgIDxIZWFkPlxuICAgICAgICA8c2NyaXB0IGRhbmdlcm91c2x5U2V0SW5uZXJIVE1MPXt7IF9faHRtbDogbW9kZVNjcmlwdCB9fSAvPlxuICAgICAgPC9IZWFkPlxuICAgICAgPGJvZHkgY2xhc3NOYW1lPVwiYmctd2hpdGUgYW50aWFsaWFzZWQgZGFyazpiZy16aW5jLTkwMFwiPlxuICAgICAgICA8TWFpbiAvPlxuICAgICAgICA8TmV4dFNjcmlwdCAvPlxuICAgICAgPC9ib2R5PlxuICAgIDwvSHRtbD5cbiAgKVxufVxuIl0sIm5hbWVzIjpbIkhlYWQiLCJIdG1sIiwiTWFpbiIsIk5leHRTY3JpcHQiLCJtb2RlU2NyaXB0IiwiRG9jdW1lbnQiLCJsYW5nIiwic2NyaXB0IiwiZGFuZ2Vyb3VzbHlTZXRJbm5lckhUTUwiLCJfX2h0bWwiLCJib2R5IiwiY2xhc3NOYW1lIl0sInNvdXJjZVJvb3QiOiIifQ==\n//# sourceURL=webpack-internal:///./src/pages/_document.jsx\n");

/***/ }),

/***/ "next/dist/compiled/next-server/pages.runtime.dev.js":
/*!**********************************************************************!*\
  !*** external "next/dist/compiled/next-server/pages.runtime.dev.js" ***!
  \**********************************************************************/
/***/ ((module) => {

module.exports = require("next/dist/compiled/next-server/pages.runtime.dev.js");

/***/ }),

/***/ "react":
/*!************************!*\
  !*** external "react" ***!
  \************************/
/***/ ((module) => {

module.exports = require("react");

/***/ }),

/***/ "react/jsx-dev-runtime":
/*!****************************************!*\
  !*** external "react/jsx-dev-runtime" ***!
  \****************************************/
/***/ ((module) => {

module.exports = require("react/jsx-dev-runtime");

/***/ }),

/***/ "path":
/*!***********************!*\
  !*** external "path" ***!
  \***********************/
/***/ ((module) => {

module.exports = require("path");

/***/ })

};
;

// load runtime
var __webpack_require__ = require("../webpack-runtime.js");
__webpack_require__.C(exports);
var __webpack_exec__ = (moduleId) => (__webpack_require__(__webpack_require__.s = moduleId))
var __webpack_exports__ = __webpack_require__.X(0, ["vendor-chunks/next","vendor-chunks/@swc"], () => (__webpack_exec__("./src/pages/_document.jsx")));
module.exports = __webpack_exports__;

})();