import React from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './NavigationHelper.module.css';

const NavigationHelper = () => {
  const { siteConfig } = useDocusaurusContext();

  return (
    <div className={styles.navigationHelper}>
      <div className={styles.quickLinks}>
        <h3>Quick Navigation</h3>
        <ul>
          <li>
            <Link to={useBaseUrl('/docs/intro')}>
              Book Introduction
            </Link>
          </li>
          <li>
            <Link to={useBaseUrl('/docs/category/embodied-ai')}>
              Embodied AI
            </Link>
          </li>
          <li>
            <Link to={useBaseUrl('/docs/category/humanoid-robotics')}>
              Humanoid Robotics
            </Link>
          </li>
          <li>
            <Link to={useBaseUrl('/docs/category/ai-integration')}>
              AI Integration
            </Link>
          </li>
        </ul>
      </div>
      <div className={styles.progressTracker}>
        <h3>Learning Path</h3>
        <p>Follow our recommended progression through the book:</p>
        <ol>
          <li><Link to={useBaseUrl('/docs/intro')}>Introduction</Link></li>
          <li><Link to={useBaseUrl('/docs/embodied-ai/introduction')}>Embodied AI</Link></li>
          <li><Link to={useBaseUrl('/docs/humanoid-robotics/design-principles')}>Humanoid Robotics</Link></li>
          <li><Link to={useBaseUrl('/docs/ai-integration/ml-locomotion')}>AI Integration</Link></li>
          <li><Link to={useBaseUrl('/docs/deployment/real-world-deployment')}>Deployment</Link></li>
        </ol>
      </div>
    </div>
  );
};

export default NavigationHelper;