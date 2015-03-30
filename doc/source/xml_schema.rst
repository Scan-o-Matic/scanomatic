XML Schema
==========

The output of the analysis is available as XML files for compatibility should
there be an interest in developing tools without needing python support.
If the possibility of using python and the ``numpy`` package exists, these
outputs should be preferred.

There is a short tag-name and a long tag-name version of otherwise identical
schemas.
The parser does not need to be a full XML parser as the output follows a strict
sequential format.
Typically the shortened tag names should be preferred as they are read much
faster.

Schema
------

Presented here is the schema for the short tags, for the long tags used in
``analysis.xml`` the exact same schema applies but with human readable names
for tags.
The translation is presented under the section where the tags are explained and
the extra restrictions within the schema clarified.

Schema short and slimmed schema for ``analysis_slimmed.xml``::
    
    <xs:element name="project">
        <xs:complexType>
            <xs:sequence>
                <xs:element name="ver" type="xs:decimal"/>
                <xs:element name="mac" type="xs:string"/>
                <xs:element name="start-t" type="xs:decimal"/>
                <xs:element name="pref" type="xs:string"/>
                <xs:element name="ptag" type="xs:string"/>
                <xs:element name="sltag" type="xs:string"/>
                <xs:element name="desc" type="xs:string"/>
                <xs:element name="n-scans" type="xs:integer"/>
                <xs:element name="int-t" type="xs:decimal"/>
                <xs:element name="n-plates" type="xs:integer"/>
                <xs:element name="matrices">
                    <xs:complexType>
                        <xs:sequence>
                            <xs:element name="p-m" type="xs:string">
                                <xs:complexType>
                                    <xs:attribute name="i" type="xs:integer"/> 
                                </xs:complexType>
                            </xs:element>
                        </xs:sequence>
                    <xs:complexType>
                </xs:element>
                <xs:element name="d-types">
                    <xs:complexType>
                        <xs:sequence>
                            <xs:element name="d-type">
                                <xs:complexType>
                                    <xs:attribute name="m" type="xs:string"/> 
                                    <xs:attribute name="u" type="xs:string"/> 
                                    <xs:attribute name="t" type="xs:string"/> 
                                </xs:complexType>
                            </xs:element>
                        </xs:sequence>
                    <xs:complexType>
                </xs:element>
                <xs:element name="compartments">
                    <xs:complexType>
                        <xs:sequence>
                            <xs:element name="compartment" type="xs:string">
                        </xs:sequence>
                    <xs:complexType>
                </xs:element>
                <xs:element name="scans">
                    <xs:complexType>
                        <xs:sequence>
                            <xs:element name="s">
                                <xs:complexType>
                                    <xs:attribute name="i" type="xs:integer"/> 
                                    <xs:sequence>
                                        <xs:element name="ok" type="xs:integer">
                                            <xs:simpleType>
                                                <xs:restriction base="xs:integer">
                                                    <xs:pattern value="[0-9]"/>
                                                </xs:restriction>
                                            </xs:simpleType>
                                        </xs:element> 
                                        <xs:element name="cal" type="xs:string"/>
                                        <xs:element name="t" type="xs:decimal"/>
                                        <xs:element name="pls">
                                            <xs:complexType>
                                                <xs:sequence>
                                                    <xs:element name="p">
                                                        <xs:complexType>
                                                            <xs:attribute name="i" type="xs:integer"/> 
                                                            <xs:sequence>
                                                                <xs:element name="pm" type="xs:string"/>
                                                                <xs:element name="gcs">
                                                                    <xs:complexType>
                                                                        <xs:sequence>
                                                                            <xs:element name="gc">
                                                                                <xs:complexType>
                                                                                    <xs:attribute name="x" type="xs:integer"/> 
                                                                                    <xs:attribute name="y" type="xs:integer"/> 
                                                                                    <xs:sequence>
                                                                                        <xs:element name="cl">
                                                                                            <xs:complexType>
                                                                                                <xs:sequence>
                                                                                                    <xs:element name="a" type="xs:float"/>
                                                                                                    <xs:element name="ps" type="xs:float"/>
                                                                                                    <xs:element name="md" type="xs:float"/>
                                                                                                    <xs:element name="IRQ" type="xs:string"/>
                                                                                                    <xs:element name="IRQ_m" type="xs:float"/>
                                                                                                    <xs:element name="m" type="xs:float"/>
                                                                                                </xs:sequence>
                                                                                            <xs:complexType>
                                                                                        </xs:element>
                                                                                        <xs:element name="bl">
                                                                                            <xs:complexType>
                                                                                                <xs:sequence>
                                                                                                    <xs:element name="per" type="xs:string"/>
                                                                                                    <xs:element name="a" type="xs:float"/>
                                                                                                    <xs:element name="ps" type="xs:float"/>
                                                                                                    <xs:element name="md" type="xs:float"/>
                                                                                                    <xs:element name="IRQ" type="xs:string"/>
                                                                                                    <xs:element name="IRQ_m" type="xs:float"/>
                                                                                                    <xs:element name="cent" type="xs:string"/>
                                                                                                    <xs:element name="m" type="xs:float"/>
                                                                                                </xs:sequence>
                                                                                            <xs:complexType>
                                                                                        </xs:element>
                                                                                        <xs:element name="bg">
                                                                                            <xs:complexType>
                                                                                                <xs:sequence>
                                                                                                    <xs:element name="a" type="xs:float"/>
                                                                                                    <xs:element name="ps" type="xs:float"/>
                                                                                                    <xs:element name="md" type="xs:float"/>
                                                                                                    <xs:element name="IRQ" type="xs:string"/>
                                                                                                    <xs:element name="IRQ_m" type="xs:float"/>
                                                                                                    <xs:element name="m" type="xs:float"/>
                                                                                                </xs:sequence>
                                                                                            <xs:complexType>
                                                                                        </xs:element>
                                                                                    </xs:sequence>
                                                                                <xs:complexType>
                                                                            </xs:element>
                                                                        </xs:sequence>
                                                                    <xs:complexType>
                                                                </xs:element>
                                                            </xs:sequence>
                                                        <xs:complexType>
                                                    </xs:element>
                                                </xs:sequence>
                                            <xs:complexType>
                                        </xs:element>
                                    </xs:sequence>
                                <xs:complexType>
                            </xs:element>
                        </xs:sequence>
                    <xs:complexType>
                </xs:element>
            </xs:sequence>
        <xs:complexType>
    </xs:element>

Tags explained
--------------

For each tag, the header shows ``short : long`` name.
Some tags are omitted as they are purely there for maintaining hierarchical
structure, and their names should be understandable.

For the ``scans`` tag, it has the restriction of always being the last tag
within the ``project`` tag. 
This is important for sequential reading as the preceding tags will have
specified the expectations on the contents of ``scans`` so that it can be
rapidly read in sequence.

ver : version
.............

The version of Scan-o-Matic that produced the output

mac : computer-mac
..................

The MAC address of the computer that scanned the project if the MAC was
obtainable through python interfaces.
Else a generated, unique but fake MAC address for that computer.

start-t : start-time
....................

Time for start of scanning in unix-time

pref : prefix
-------------

Project prefix, should match the name of the folder in which the project was
placeed.

ptag : project_tag
..................

Project tag.
Used for tracking experiments by accessory program.
See ``sltag``.

sltag : scanner_layout_tag
..........................

Scan layout tag.
Used for tracking experiments by accessory program.
Scan layout is implied as a unique tag for the specific plates used in the
specific current experiment, which can be part of a larger project (``ptag``).
These tags should follow a specific hex format and be used with a checksum
value.
Details for their implementation and generation of checksum can be found in
``scanomatic.io.verificationTags``.

desc : description
..................

Free text description of the plates scanned and the current project

n-scans : number-of-scans
.........................

Number of scanns in the project

**This invokes a restriction on the exact number of ``s`` tags needed as well as
their attribute ``i``.**

int-t : interval-time
.....................

The interval between scans in minutes


n-plates : plates-per-scan
..........................

The number of plates in the fixture in this project

**This invokes a restriction on the exact number of ``pl`` in ``pls``**

p-m : pinning-matrix
....................

Pinning matrix for each plate.
Index ``i``/``index`` attribute indicate position in fixture.
Value is a tuple of integers indicating number of colonies along each dimension
of the plate.

**This invokes a restriction on exact number of ``gc`` in each ``gcs`` as well
as their attributes ``x`` and ``y``**

d-types : d-types
.................

List the types of measures allowed for each grid cell/colony of the matrix.

All measures must not exist on all compartments of each grid cell, but only
measured listed are allowed.

d-type : d-type
...............

Data type/a measure.

Attribute ``m``/``measure`` specifies the tag name of the measure.

Attribute ``u``/``unit`` specifies the unit of the measure.

Attribute ``t``/``type`` specifies the type of measure, where ``standard`` implies a
value known to XML and specified in the schema.
Other values are typically descriptions of complex types such as lists.

compartment : compartment
.........................

Specified the logical compartments analysed in each grid cells.
They can be:

    * ``cell``: The entire grid cell segment
    * ``blob``: The segment of the grid cell determined to house the feature of
      interest (the colony).
    * ``background``: The compliment of the ``blob`` minus a buffe zone.

**This invokes restriction on what tags are allowed in the ``gc``**

s : scan
........

The tag collecting all information pertaining to a specific scan.

The attribute ``i``/``index`` is the number in sequence of scans the current scan data
refers to.
Allowed values is restricted by the ``n-scans``.

ok : scan-valid
...............

Integer specifying if the grayscale was understood and well behaved (``1``).
If not (``0``) no analysis and no further tags will be allowed.

cal : calibration
.................

Calibration values for the grayscale

t : time
........

Relative time since start of experiment for when the image was taken.

pls : plates
............

Collection of the plates
Number of plates within this tag restricted by ``n-plates``.

p : plate
.........

Collection of tags pertaining to a plate in fixture position indicated by
attribute ``i``/``index``.
Note that fixture positions are enumerated from 0, this is only translated to
enumeration from 1 in the user interface.

pm : pinning-matrix
...................

Pinning matrix used for the plate.

**Deprecated. Use global settings in ``matrices`` instead**

gcs : grid-cells
................

Collection of the grid cells on the plate

gc : grid-cell
..............

The grid cell.
Its position in the pinning matrix specified by the attributes ``x`` and ``y``.
These are restricted by the global pinning matrix.

cl : cell
.........

Measures for the entire grid cell.
These are only allowed and present if this segment was specified in the global
``compartments`` settings.

bl : blob
.........

Measures for the blob (colony)
These are only allowed and present if this segment was specified in the global
``compartments`` settings.

bg : background
...............

Measures for the background.
These are only allowed and present if this segment was specified in the global
``compartments`` settings.

per : perimeter
...............

Perimeter.
Only valid for the blob segment.

**Not implemented**

a : area
........

The area of the segment.

ps : pixelsum
.............

The sum of the pixels values.

(Pixel values are typically as cells per pixels)

md : median
...........

Median.

(Typically as cells per pixels)

IRQ : IRQ
.........

Lower and upper bound of inter quartile range.

(Typically as cells per pixels)

IRQ_m : IRQ_mean
................

Mean of the inter quartile range.

(Typically as cells per pixels)

cent : centroid
...............

Center of mass.
Only valid for blob segment.
The X and Y coordinate of the blob center in the coordinate system of the grid
cell.

m : mean
........

Mean

(Typically as cells per pixels)
