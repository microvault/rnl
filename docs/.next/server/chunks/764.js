"use strict";exports.id=764,exports.ids=[764],exports.modules={6764:(e,t,d)=>{d.r(t),d.d(t,{default:()=>Document});var o=d(997),i=d(6859);let a=`
  let darkModeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)')

  updateMode()
  darkModeMediaQuery.addEventListener('change', updateModeWithoutTransitions)
  window.addEventListener('storage', updateModeWithoutTransitions)

  function updateMode() {
    let isSystemDarkMode = darkModeMediaQuery.matches
    let isDarkMode = window.localStorage.isDarkMode === 'true' || (!('isDarkMode' in window.localStorage) && isSystemDarkMode)

    if (isDarkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }

    if (isDarkMode === isSystemDarkMode) {
      delete window.localStorage.isDarkMode
    }
  }

  function disableTransitionsTemporarily() {
    document.documentElement.classList.add('[&_*]:!transition-none')
    window.setTimeout(() => {
      document.documentElement.classList.remove('[&_*]:!transition-none')
    }, 0)
  }

  function updateModeWithoutTransitions() {
    disableTransitionsTemporarily()
    updateMode()
  }
`;function Document(){return(0,o.jsxs)(i.Html,{lang:"en",children:[o.jsx(i.Head,{children:o.jsx("script",{dangerouslySetInnerHTML:{__html:a}})}),(0,o.jsxs)("body",{className:"bg-white antialiased dark:bg-zinc-900",children:[o.jsx(i.Main,{}),o.jsx(i.NextScript,{})]})]})}}};