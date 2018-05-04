import React from 'react';
import { storiesOf } from '@storybook/react';
import { action } from '@storybook/addon-actions';
import ProjectPanel from './ProjectPanel';
import '../../../style/bootstrap.css';
import '../../../style/project.css';


storiesOf('ProjectPanel', module)
    .addDecorator(story => (
        <div className="row">
            <div className="col-md-offset-1 col-md-10">
                {story()}
            </div>
        </div>
    ))
    .add('default collapsed view', () => (
        <ProjectPanel
            name="I am project"
            description="The rapid horizontal transmission of many antibiotic resistance genes between bacterial host cells on conjugative plasmids is a major cause of the accelerating antibiotic resistance crisis. Preventing understanding and targeting conjugation, there currently are no experimental platforms for fast and cost-efficient screening of genetic effects on antibiotic resistance transmission by conjugation. We introduce a novel experimental framework to screen for conjugation based horizontal transmission of antibiotic resistance between >60,000 pairs of cell populations in parallel. Plasmid-carrying donor strains are constructed in high throughput."
            onNewExperiment={action('newExperiment')}
        />
    ))
    .add('disable "New Experiment"', () => (
        <ProjectPanel
            name="I am project"
            description="The rapid horizontal transmission of many antibiotic resistance genes between bacterial host cells on conjugative plasmids is a major cause of the accelerating antibiotic resistance crisis. Preventing understanding and targeting conjugation, there currently are no experimental platforms for fast and cost-efficient screening of genetic effects on antibiotic resistance transmission by conjugation. We introduce a novel experimental framework to screen for conjugation based horizontal transmission of antibiotic resistance between >60,000 pairs of cell populations in parallel. Plasmid-carrying donor strains are constructed in high throughput."
            onNewExperiment={action('newExperiment')}
            newExperimentDisabled
        />
    ))
    .add('with children', () => (
        <ProjectPanel
            name="I am project"
            description="The rapid horizontal transmission of many antibiotic resistance genes between bacterial host cells on conjugative plasmids is a major cause of the accelerating antibiotic resistance crisis. Preventing understanding and targeting conjugation, there currently are no experimental platforms for fast and cost-efficient screening of genetic effects on antibiotic resistance transmission by conjugation. We introduce a novel experimental framework to screen for conjugation based horizontal transmission of antibiotic resistance between >60,000 pairs of cell populations in parallel. Plasmid-carrying donor strains are constructed in high throughput."
            onNewExperiment={action('newExperiment')}
        >
            <div>I am a child!</div>
            <div>Me too!</div>
        </ProjectPanel>
    ));
