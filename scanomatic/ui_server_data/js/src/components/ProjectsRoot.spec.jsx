import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ProjectsRoot from './ProjectsRoot';

describe('<ProjectsRoot />', () => {
    const props = {
        projects: [],
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
            },
            {
                id: '2',
                name: 'Old Test',
                description: 'Bla bala bal',
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

    it('should bing the <ProjectPanel/> onNewExperiment callback to the project id', () => {
        const projects = [{ id: '147', name: 'Foo', description: 'bar' }];
        const onNewExperiment = jasmine.createSpy('onNewExperiment');
        const wrapper = shallow(<ProjectsRoot {...props} projects={projects} onNewExperiment={onNewExperiment} />);
        wrapper.find('ProjectPanel').prop('onNewExperiment')();
        expect(onNewExperiment).toHaveBeenCalledWith('147');
    });
});
