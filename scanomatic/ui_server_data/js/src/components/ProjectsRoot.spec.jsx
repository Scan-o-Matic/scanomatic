import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ProjectsRoot from './ProjectsRoot';

describe('<ProjectsRoot />', () => {
    const props = {
        projects: [],
        experimentActions: {
            onStart: () => {},
            onRemove: () => {},
            onStop: () => {},
            onDone: () => {},
            onReopen: () => {},
            onFeatureExtract: () => {},
        },
        newExperimentActions: {
            onChange: () => {},
            onCancel: () => {},
            onSubmit: () => {},
        },
        newProject: null,
        newProjectActions: {
            onChange: () => {},
            onCancel: () => {},
            onSubmit: () => {},
        },
        onNewProject: () => {},
        onNewExperiment: () => {},
    };

    describe('New Project button', () => {
        let btn;
        const onNewProject = jasmine.createSpy('onNewProject');

        beforeEach(() => {
            onNewProject.calls.reset();
            const wrapper = shallow(<ProjectsRoot {...props} onNewProject={onNewProject} />);
            btn = wrapper.find('.new-project');
        });

        it('renders', () => {
            expect(btn.exists()).toBeTruthy();
            expect(btn.text()).toEqual(' New Project');
            expect(btn.hasClass('btn')).toBeTruthy();
        });

        it('calls onNewProject when clicked', () => {
            btn.simulate('click');
            expect(onNewProject).toHaveBeenCalled();
        });
    });

    describe('<NewProjectPanel />', () => {
        let form;
        const newProject = {
            name: 'Yeast has a taste for music',
            description: 'Ignobel here I come!',
        };
        const newProjectActions = {
            error: 'Volume too low...',
            onSubmit: () => {},
            onChange: () => {},
            onCancel: () => {},
        };

        beforeEach(() => {
            const wrapper = shallow(<ProjectsRoot
                {...props}
                newProject={newProject}
                newProjectActions={newProjectActions}
            />);
            form = wrapper.find('NewProjectPanel');
        });

        it('renders', () => {
            expect(form.exists()).toBeTruthy();
        });

        it('passes name and description', () => {
            expect(form.prop('name')).toEqual(newProject.name);
            expect(form.prop('description')).toEqual(newProject.description);
        });

        it('passes the actions and error', () => {
            expect(form.prop('error')).toEqual(newProjectActions.error);
            expect(form.prop('onSubmit')).toBe(newProjectActions.onSubmit);
            expect(form.prop('onChange')).toBe(newProjectActions.onChange);
            expect(form.prop('onCancel')).toBe(newProjectActions.onCancel);
        });
    });

    it('does not render a new project form', () => {
        const wrapper = shallow(<ProjectsRoot {...props} />);
        const form = wrapper.find('NewProjectPanel');
        expect(form.exists()).toBeFalsy();
    });

    describe('projects', () => {
        const projects = [
            {
                id: '1',
                name: 'Test',
                description: 'Testing',
                experiments: [],
            },
            {
                id: '2',
                name: 'Old Test',
                description: 'Bla bala bal',
                experiments: [],
            },
        ];

        let projectPanels;

        beforeEach(() => {
            const wrapper = shallow(<ProjectsRoot {...props} projects={projects} />);
            projectPanels = wrapper.find('ProjectPanel');
        });

        it('displays two panels', () => {
            expect(projectPanels.length).toEqual(2);
            expect(projectPanels.at(0).key()).toEqual(projects[0].name);
            expect(projectPanels.at(1).key()).toEqual(projects[1].name);
        });

        it('passes name and description', () => {
            expect(projectPanels.at(0).prop('name')).toEqual(projects[0].name);
            expect(projectPanels.at(1).prop('name')).toEqual(projects[1].name);
            expect(projectPanels.at(0).prop('description')).toEqual(projects[0].description);
            expect(projectPanels.at(1).prop('description')).toEqual(projects[1].description);
        });
    });

    it('should bind the <ProjectPanel/> onNewExperiment callback to the project id', () => {
        const projects = [{
            id: '147', name: 'Foo', description: 'bar', experiments: [],
        }];
        const onNewExperiment = jasmine.createSpy('onNewExperiment');
        const wrapper = shallow(<ProjectsRoot
            {...props}
            projects={projects}
            onNewExperiment={onNewExperiment}
        />);
        wrapper.find('ProjectPanel').prop('onNewExperiment')();
        expect(onNewExperiment).toHaveBeenCalledWith('147');
    });

    it(
        'should create <NewExperimentPanel/> under <ProjectPanel/> with corresponding id',
        () => {
            const projects = [
                {
                    id: '42', name: 'Foo', description: 'bar', experiments: [],
                },
                {
                    id: '147', name: 'Bla', description: 'Bla bla bl', experiments: [],
                },
            ];
            const newExperiment = {
                name: 'grow stuff',
                description: '',
                duration: 123,
                interval: 1,
                scannerId: '',
                projectId: '42',
                pinning: new Map([[1, null]]),
            };
            const wrapper = shallow(<ProjectsRoot
                {...props}
                projects={projects}
                newExperiment={newExperiment}
            />);
            expect(wrapper.find('ProjectPanel[id="42"]').find('NewExperimentPanel').exists())
                .toBeTruthy();
            expect(wrapper.find('ProjectPanel[id="147"]').find('NewExperimentPanel').exists())
                .toBeFalsy();
        },
    );

    it(
        'should pass newExperimentDisabled=true to <ProjectPanel/> if there is a new experiment for project',
        () => {
            const projects = [
                {
                    id: '42', name: 'Foo', description: 'bar', experiments: [],
                },
            ];
            const newExperiment = {
                name: 'grow stuff',
                description: '',
                duration: 123,
                interval: 1,
                scannerId: '',
                projectId: '42',
                pinning: new Map([[1, null]]),
            };
            const wrapper = shallow(<ProjectsRoot
                {...props}
                projects={projects}
                newExperiment={newExperiment}
            />);
            expect(wrapper.find('ProjectPanel[id="42"]').prop('newExperimentDisabled')).toBeTruthy();
        },
    );
    describe('experiments', () => {
        const projects = [
            {
                id: '42',
                name: 'Foo',
                description: 'bar',
                experiments: [{
                    id: '01',
                    name: 'First Experiment',
                    description: 'Bla bla',
                    duration: 36000000,
                    interval: 500000,
                    pinning: new Map([[1: ''], [2: '384'], [3: ''], [4: '']]),
                    scanner: {
                        identifier: 'S01', name: 'Scanny', power: true, owned: true,
                    },
                }],
            },
        ];
        let wrapper;

        beforeEach(() => {
            wrapper = shallow(<ProjectsRoot {...props} projects={projects} />);
        });

        it(
            'should create an <ExperimentPanel/> under <ProjectPanel/> for each experiment',
            () => {
                expect(wrapper.find('ProjectPanel[id="42"]').find('ExperimentPanel').exists())
                    .toBeTruthy();
            },
        );

        it('should pass experiment action onStart', () => {
            expect(wrapper
                .find('ProjectPanel[id="42"]')
                .find('ExperimentPanel')
                .prop('onStart')).toEqual(props.experimentActions.onStart);
        });

        it('should pass experiment action onRemove', () => {
            expect(wrapper
                .find('ProjectPanel[id="42"]')
                .find('ExperimentPanel')
                .prop('onRemove')).toEqual(props.experimentActions.onRemove);
        });
    });
});
