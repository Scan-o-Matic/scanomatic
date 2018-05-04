import React from 'react';
import { storiesOf } from '@storybook/react';
import ExperimentPanel from './index';
import '../../../../style/bootstrap.css';
import '../../../../style/project.css';


storiesOf('ExperimentPanel', module)
    .addDecorator(story => (
        <div className="row">
            <div className="col-md-offset-1 col-md-10">
                {story()}
            </div>
        </div>
    ))
    .add('Planned', () => (
        <ExperimentPanel
            name="I am testing"
            description="If you are considering setting up Scan-o-matic at your lab, we would be very happy and would love to hear from you. But, before you decide on this, the Faculty of Science at University of Gothenburg has included Scan-o-matic among its high-throughput phenomics infrastructure and it is our expressed interest that external researchers come to us. If you are interested there's some more information and contact information here: The center for large scale cell based screeening. It is yet to become listed on the page, but don't worry, it will be part of the list."
            duration={1200000}
            interval={600000}
            scanner={{
                id: 'myscannerid', name: 'Fake scanner', owned: false, power: true,
            }}
        />
    ))
    .add('Running', () => (
        <ExperimentPanel
            name="I am testing"
            description="If you are considering setting up Scan-o-matic at your lab, we would be very happy and would love to hear from you. But, before you decide on this, the Faculty of Science at University of Gothenburg has included Scan-o-matic among its high-throughput phenomics infrastructure and it is our expressed interest that external researchers come to us. If you are interested there's some more information and contact information here: The center for large scale cell based screeening. It is yet to become listed on the page, but don't worry, it will be part of the list."
            duration={1200000}
            interval={600000}
            scanner={{
                id: 'myscannerid', name: 'Fake scanner', owned: false, power: true,
            }}
            started={new Date()}
            end={new Date(new Date().getTime() + 1200000)}
        />
    ));
