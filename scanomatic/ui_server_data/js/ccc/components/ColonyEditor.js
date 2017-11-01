import PropTypes from 'prop-types';
import React from 'react';

import ColonyFeatures from './ColonyFeatures';
import ColonyImage from './ColonyImage';

export default class ColonyEditor extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            drawing: false,
            data: props.data,
        }
        this.handleClickFix = this.handleClickFix.bind(this);
        this.handleUpdate = this.handleUpdate.bind(this);
    }

    componentWillReceiveProps(props) {
        this.setState({
            drawing: false,
            data: props.data,
        });
    }

    handleClickFix() {
        this.setState({ drawing: true });
    }

    handleUpdate(data) {
        this.setState({ drawing: false });
        this.props.onUpdate && this.props.onUpdate(data);
    }

    render() {
        const style = {
            background: 'white',
            display: 'inline-block',
            paddingd: '3px',
            textAlign: 'center',
        };

        return (
            <div>
                <div><span>Colony Image</span></div>
                <ColonyImage
                    data={this.state.data}
                    draw={this.state.drawing}
                    onUpdate={this.handleUpdate}
                />
                <div><br /></div>
                <div><span>Colony MetaData</span></div>
                <div style={style} >
                    <ColonyFeatures data={this.state.data} />
                </div>
                <button
                    className="btn btn-default btn-fix"
                    style={{ horizAlign: 'center' }}
                    onClick={this.handleClickFix}
                >Fix</button>
                <div style={{ textAlign: 'center' }}>
                    <button
                        className="btn btn-default btn-set"
                        style={{ marginLeft: '30px' }}
                        onClick={this.props.onSet}
                    >Set</button>
                </div>
            </div>
        );
    }
}

ColonyEditor.propTypes = {
    data: PropTypes.object.isRequired,
    onFix: PropTypes.func,
    onSet: PropTypes.func,
    onUpdate: PropTypes.func,
};
