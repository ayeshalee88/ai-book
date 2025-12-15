import React from 'react';
import { Navbar } from '@docusaurus/theme-classic';

const Layout = ({ children }) => {
  return (
    <>
      <Navbar />
      <main>{children}</main>
    </>
  );
};

export default Layout;