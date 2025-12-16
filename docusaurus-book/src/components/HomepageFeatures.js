import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Learning Outcomes',
    imageUrl: require('../../static/img/aye.webp').default,
    description: (
      <>
        Master Physical AI principles, embodied intelligence,
        humanoid robotics, and conversational AI through
        our structured 13-week curriculum.
      <br />
        <a href="/docs/course-overview/learning-outcomes">Explore outcomes →</a>
      </>
    ),
  },
  {
    title: 'Humanoid Robotics',
    imageUrl: require('../../static/img/bada.png').default,
    description: (
      <>
        Explore the design principles, kinematics, and control systems
        behind humanoid robots that bridge digital and physical intelligence.
      </>
    ),
  },
  {
    title: 'Practical Implementation',
    imageUrl: require('../../static/img/ammi.png').default,
    description: (
      <>
        Get hands-on experience with tutorials, code examples,
        and real-world case studies from industry leaders.
      </>
    ),
  },
];

function Feature({imageUrl, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <img src={imageUrl} className={styles.featureSvg} alt={title} />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}