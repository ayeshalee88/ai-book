// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Bridging the gap between digital AI and physical robotics',
  favicon: 'img/favicon.ico',

  url: 'https://ayeshalee88.github.io',
  baseUrl: '/',

  organizationName: 'ayeshalee88',
  projectName: 'ai-book',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/ayeshalee88/ai-book/edit/main/docusaurus-book/',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',

    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Physical AI Book Logo',
        src: 'img/badam.jpg',
        srcDark: 'img/badam.jpg',
        width: 50,
        height: 100,
        style: {
          borderRadius: '50%',
          overflow: 'hidden',
          border: '3px solid #0800ffd5',
          boxShadow: '0 0 15px rgba(30, 0, 255, 0.11)',
        },
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Book',
        },
        {
          href: 'https://github.com/ayeshalee88/ai-book',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    search: {
      provider: 'local',
    },

    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },

    prism: {
      theme: lightCodeTheme,
      darkTheme: darkCodeTheme,
      additionalLanguages: ['python', 'bash', 'json', 'yaml', 'robotframework'],
    },

    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },

    footer: {
      style: 'dark',
      links: [
        {
          title: 'Book Sections',
          items: [
            { label: 'Introduction', to: '/docs/intro' },
            { label: 'Embodied AI', to: '/docs/embodied-ai/introduction' },
            { label: 'Humanoid Robotics', to: '/docs/humanoid-robotics/design-principles' },
          ],
        },
        {
          title: 'Community',
          items: [
            { label: 'Stack Overflow', href: 'https://stackoverflow.com/questions/tagged/docusaurus' },
            { label: 'Discord', href: 'https://discordapp.com/invite/docusaurus' },
            { label: 'Twitter', href: 'https://twitter.com/docusaurus' },
          ],
        },
        {
          title: 'More',
          items: [
            { label: 'GitHub', href: 'https://github.com/ayeshalee88' },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Book. Built with Docusaurus. Powered by Ayisha`,
    },
  },
plugins: [
  function webpackPolyfillPlugin() {
    return {
      name: 'webpack-polyfill-plugin',
      configureWebpack() {
        const webpack = require('webpack');
        return {
          resolve: {
            fallback: {
              fs: false,
              path: require.resolve('path-browserify'),
              stream: require.resolve('stream-browserify'),
              util: require.resolve('util/'),
              crypto: require.resolve('crypto-browserify'),
              os: require.resolve('os-browserify/browser'),
              buffer: require.resolve('buffer/'),
              constants: require.resolve('constants-browserify'),
              url: require.resolve('url/'),
              module: false,
              child_process: false,
              process: require.resolve('process/browser.js'),
            },
          },
          plugins: [
            new webpack.ProvidePlugin({
              Buffer: ['buffer', 'Buffer'],
              process: 'process/browser.js',
            }),
          ],
        };
      },
    };
  },
]

  
    
    
};

module.exports = config;
