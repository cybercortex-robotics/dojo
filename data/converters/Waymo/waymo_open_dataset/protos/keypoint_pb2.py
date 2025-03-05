# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: waymo_open_dataset/protos/keypoint.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='waymo_open_dataset/protos/keypoint.proto',
  package='waymo.open_dataset.keypoints',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n(waymo_open_dataset/protos/keypoint.proto\x12\x1cwaymo.open_dataset.keypoints\"\x1d\n\x05Vec2d\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\"(\n\x05Vec3d\x12\t\n\x01x\x18\x01 \x01(\x01\x12\t\n\x01y\x18\x02 \x01(\x01\x12\t\n\x01z\x18\x03 \x01(\x01\")\n\x12KeypointVisibility\x12\x13\n\x0bis_occluded\x18\x01 \x01(\x08\"\x97\x01\n\nKeypoint2d\x12\x38\n\x0blocation_px\x18\x01 \x01(\x0b\x32#.waymo.open_dataset.keypoints.Vec2d\x12\x44\n\nvisibility\x18\x02 \x01(\x0b\x32\x30.waymo.open_dataset.keypoints.KeypointVisibility*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02\"\x96\x01\n\nKeypoint3d\x12\x37\n\nlocation_m\x18\x01 \x01(\x0b\x32#.waymo.open_dataset.keypoints.Vec3d\x12\x44\n\nvisibility\x18\x02 \x01(\x0b\x32\x30.waymo.open_dataset.keypoints.KeypointVisibility*\t\x08\xe8\x07\x10\x80\x80\x80\x80\x02\"\xc8\x01\n\x0e\x43\x61meraKeypoint\x12\x38\n\x04type\x18\x01 \x01(\x0e\x32*.waymo.open_dataset.keypoints.KeypointType\x12=\n\x0bkeypoint_2d\x18\x02 \x01(\x0b\x32(.waymo.open_dataset.keypoints.Keypoint2d\x12=\n\x0bkeypoint_3d\x18\x03 \x01(\x0b\x32(.waymo.open_dataset.keypoints.Keypoint3d\"Q\n\x0f\x43\x61meraKeypoints\x12>\n\x08keypoint\x18\x01 \x03(\x0b\x32,.waymo.open_dataset.keypoints.CameraKeypoint\"\x88\x01\n\rLaserKeypoint\x12\x38\n\x04type\x18\x01 \x01(\x0e\x32*.waymo.open_dataset.keypoints.KeypointType\x12=\n\x0bkeypoint_3d\x18\x02 \x01(\x0b\x32(.waymo.open_dataset.keypoints.Keypoint3d\"O\n\x0eLaserKeypoints\x12=\n\x08keypoint\x18\x01 \x03(\x0b\x32+.waymo.open_dataset.keypoints.LaserKeypoint*\xee\x03\n\x0cKeypointType\x12\x1d\n\x19KEYPOINT_TYPE_UNSPECIFIED\x10\x00\x12\x16\n\x12KEYPOINT_TYPE_NOSE\x10\x01\x12\x1f\n\x1bKEYPOINT_TYPE_LEFT_SHOULDER\x10\x05\x12\x1c\n\x18KEYPOINT_TYPE_LEFT_ELBOW\x10\x06\x12\x1c\n\x18KEYPOINT_TYPE_LEFT_WRIST\x10\x07\x12\x1a\n\x16KEYPOINT_TYPE_LEFT_HIP\x10\x08\x12\x1b\n\x17KEYPOINT_TYPE_LEFT_KNEE\x10\t\x12\x1c\n\x18KEYPOINT_TYPE_LEFT_ANKLE\x10\n\x12 \n\x1cKEYPOINT_TYPE_RIGHT_SHOULDER\x10\r\x12\x1d\n\x19KEYPOINT_TYPE_RIGHT_ELBOW\x10\x0e\x12\x1d\n\x19KEYPOINT_TYPE_RIGHT_WRIST\x10\x0f\x12\x1b\n\x17KEYPOINT_TYPE_RIGHT_HIP\x10\x10\x12\x1c\n\x18KEYPOINT_TYPE_RIGHT_KNEE\x10\x11\x12\x1d\n\x19KEYPOINT_TYPE_RIGHT_ANKLE\x10\x12\x12\x1a\n\x16KEYPOINT_TYPE_FOREHEAD\x10\x13\x12\x1d\n\x19KEYPOINT_TYPE_HEAD_CENTER\x10\x14')
)

_KEYPOINTTYPE = _descriptor.EnumDescriptor(
  name='KeypointType',
  full_name='waymo.open_dataset.keypoints.KeypointType',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='KEYPOINT_TYPE_UNSPECIFIED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='KEYPOINT_TYPE_NOSE', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='KEYPOINT_TYPE_LEFT_SHOULDER', index=2, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='KEYPOINT_TYPE_LEFT_ELBOW', index=3, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='KEYPOINT_TYPE_LEFT_WRIST', index=4, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='KEYPOINT_TYPE_LEFT_HIP', index=5, number=8,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='KEYPOINT_TYPE_LEFT_KNEE', index=6, number=9,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='KEYPOINT_TYPE_LEFT_ANKLE', index=7, number=10,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='KEYPOINT_TYPE_RIGHT_SHOULDER', index=8, number=13,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='KEYPOINT_TYPE_RIGHT_ELBOW', index=9, number=14,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='KEYPOINT_TYPE_RIGHT_WRIST', index=10, number=15,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='KEYPOINT_TYPE_RIGHT_HIP', index=11, number=16,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='KEYPOINT_TYPE_RIGHT_KNEE', index=12, number=17,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='KEYPOINT_TYPE_RIGHT_ANKLE', index=13, number=18,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='KEYPOINT_TYPE_FOREHEAD', index=14, number=19,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='KEYPOINT_TYPE_HEAD_CENTER', index=15, number=20,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1004,
  serialized_end=1498,
)
_sym_db.RegisterEnumDescriptor(_KEYPOINTTYPE)

KeypointType = enum_type_wrapper.EnumTypeWrapper(_KEYPOINTTYPE)
KEYPOINT_TYPE_UNSPECIFIED = 0
KEYPOINT_TYPE_NOSE = 1
KEYPOINT_TYPE_LEFT_SHOULDER = 5
KEYPOINT_TYPE_LEFT_ELBOW = 6
KEYPOINT_TYPE_LEFT_WRIST = 7
KEYPOINT_TYPE_LEFT_HIP = 8
KEYPOINT_TYPE_LEFT_KNEE = 9
KEYPOINT_TYPE_LEFT_ANKLE = 10
KEYPOINT_TYPE_RIGHT_SHOULDER = 13
KEYPOINT_TYPE_RIGHT_ELBOW = 14
KEYPOINT_TYPE_RIGHT_WRIST = 15
KEYPOINT_TYPE_RIGHT_HIP = 16
KEYPOINT_TYPE_RIGHT_KNEE = 17
KEYPOINT_TYPE_RIGHT_ANKLE = 18
KEYPOINT_TYPE_FOREHEAD = 19
KEYPOINT_TYPE_HEAD_CENTER = 20



_VEC2D = _descriptor.Descriptor(
  name='Vec2d',
  full_name='waymo.open_dataset.keypoints.Vec2d',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='waymo.open_dataset.keypoints.Vec2d.x', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='waymo.open_dataset.keypoints.Vec2d.y', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=74,
  serialized_end=103,
)


_VEC3D = _descriptor.Descriptor(
  name='Vec3d',
  full_name='waymo.open_dataset.keypoints.Vec3d',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='waymo.open_dataset.keypoints.Vec3d.x', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='y', full_name='waymo.open_dataset.keypoints.Vec3d.y', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='z', full_name='waymo.open_dataset.keypoints.Vec3d.z', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=105,
  serialized_end=145,
)


_KEYPOINTVISIBILITY = _descriptor.Descriptor(
  name='KeypointVisibility',
  full_name='waymo.open_dataset.keypoints.KeypointVisibility',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='is_occluded', full_name='waymo.open_dataset.keypoints.KeypointVisibility.is_occluded', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=147,
  serialized_end=188,
)


_KEYPOINT2D = _descriptor.Descriptor(
  name='Keypoint2d',
  full_name='waymo.open_dataset.keypoints.Keypoint2d',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='location_px', full_name='waymo.open_dataset.keypoints.Keypoint2d.location_px', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='visibility', full_name='waymo.open_dataset.keypoints.Keypoint2d.visibility', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=True,
  syntax='proto2',
  extension_ranges=[(1000, 536870912), ],
  oneofs=[
  ],
  serialized_start=191,
  serialized_end=342,
)


_KEYPOINT3D = _descriptor.Descriptor(
  name='Keypoint3d',
  full_name='waymo.open_dataset.keypoints.Keypoint3d',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='location_m', full_name='waymo.open_dataset.keypoints.Keypoint3d.location_m', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='visibility', full_name='waymo.open_dataset.keypoints.Keypoint3d.visibility', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=True,
  syntax='proto2',
  extension_ranges=[(1000, 536870912), ],
  oneofs=[
  ],
  serialized_start=345,
  serialized_end=495,
)


_CAMERAKEYPOINT = _descriptor.Descriptor(
  name='CameraKeypoint',
  full_name='waymo.open_dataset.keypoints.CameraKeypoint',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='waymo.open_dataset.keypoints.CameraKeypoint.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='keypoint_2d', full_name='waymo.open_dataset.keypoints.CameraKeypoint.keypoint_2d', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='keypoint_3d', full_name='waymo.open_dataset.keypoints.CameraKeypoint.keypoint_3d', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=498,
  serialized_end=698,
)


_CAMERAKEYPOINTS = _descriptor.Descriptor(
  name='CameraKeypoints',
  full_name='waymo.open_dataset.keypoints.CameraKeypoints',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='keypoint', full_name='waymo.open_dataset.keypoints.CameraKeypoints.keypoint', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=700,
  serialized_end=781,
)


_LASERKEYPOINT = _descriptor.Descriptor(
  name='LaserKeypoint',
  full_name='waymo.open_dataset.keypoints.LaserKeypoint',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='waymo.open_dataset.keypoints.LaserKeypoint.type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='keypoint_3d', full_name='waymo.open_dataset.keypoints.LaserKeypoint.keypoint_3d', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=784,
  serialized_end=920,
)


_LASERKEYPOINTS = _descriptor.Descriptor(
  name='LaserKeypoints',
  full_name='waymo.open_dataset.keypoints.LaserKeypoints',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='keypoint', full_name='waymo.open_dataset.keypoints.LaserKeypoints.keypoint', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=922,
  serialized_end=1001,
)

_KEYPOINT2D.fields_by_name['location_px'].message_type = _VEC2D
_KEYPOINT2D.fields_by_name['visibility'].message_type = _KEYPOINTVISIBILITY
_KEYPOINT3D.fields_by_name['location_m'].message_type = _VEC3D
_KEYPOINT3D.fields_by_name['visibility'].message_type = _KEYPOINTVISIBILITY
_CAMERAKEYPOINT.fields_by_name['type'].enum_type = _KEYPOINTTYPE
_CAMERAKEYPOINT.fields_by_name['keypoint_2d'].message_type = _KEYPOINT2D
_CAMERAKEYPOINT.fields_by_name['keypoint_3d'].message_type = _KEYPOINT3D
_CAMERAKEYPOINTS.fields_by_name['keypoint'].message_type = _CAMERAKEYPOINT
_LASERKEYPOINT.fields_by_name['type'].enum_type = _KEYPOINTTYPE
_LASERKEYPOINT.fields_by_name['keypoint_3d'].message_type = _KEYPOINT3D
_LASERKEYPOINTS.fields_by_name['keypoint'].message_type = _LASERKEYPOINT
DESCRIPTOR.message_types_by_name['Vec2d'] = _VEC2D
DESCRIPTOR.message_types_by_name['Vec3d'] = _VEC3D
DESCRIPTOR.message_types_by_name['KeypointVisibility'] = _KEYPOINTVISIBILITY
DESCRIPTOR.message_types_by_name['Keypoint2d'] = _KEYPOINT2D
DESCRIPTOR.message_types_by_name['Keypoint3d'] = _KEYPOINT3D
DESCRIPTOR.message_types_by_name['CameraKeypoint'] = _CAMERAKEYPOINT
DESCRIPTOR.message_types_by_name['CameraKeypoints'] = _CAMERAKEYPOINTS
DESCRIPTOR.message_types_by_name['LaserKeypoint'] = _LASERKEYPOINT
DESCRIPTOR.message_types_by_name['LaserKeypoints'] = _LASERKEYPOINTS
DESCRIPTOR.enum_types_by_name['KeypointType'] = _KEYPOINTTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Vec2d = _reflection.GeneratedProtocolMessageType('Vec2d', (_message.Message,), {
  'DESCRIPTOR' : _VEC2D,
  '__module__' : 'waymo_open_dataset.protos.keypoint_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.keypoints.Vec2d)
  })
_sym_db.RegisterMessage(Vec2d)

Vec3d = _reflection.GeneratedProtocolMessageType('Vec3d', (_message.Message,), {
  'DESCRIPTOR' : _VEC3D,
  '__module__' : 'waymo_open_dataset.protos.keypoint_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.keypoints.Vec3d)
  })
_sym_db.RegisterMessage(Vec3d)

KeypointVisibility = _reflection.GeneratedProtocolMessageType('KeypointVisibility', (_message.Message,), {
  'DESCRIPTOR' : _KEYPOINTVISIBILITY,
  '__module__' : 'waymo_open_dataset.protos.keypoint_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.keypoints.KeypointVisibility)
  })
_sym_db.RegisterMessage(KeypointVisibility)

Keypoint2d = _reflection.GeneratedProtocolMessageType('Keypoint2d', (_message.Message,), {
  'DESCRIPTOR' : _KEYPOINT2D,
  '__module__' : 'waymo_open_dataset.protos.keypoint_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.keypoints.Keypoint2d)
  })
_sym_db.RegisterMessage(Keypoint2d)

Keypoint3d = _reflection.GeneratedProtocolMessageType('Keypoint3d', (_message.Message,), {
  'DESCRIPTOR' : _KEYPOINT3D,
  '__module__' : 'waymo_open_dataset.protos.keypoint_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.keypoints.Keypoint3d)
  })
_sym_db.RegisterMessage(Keypoint3d)

CameraKeypoint = _reflection.GeneratedProtocolMessageType('CameraKeypoint', (_message.Message,), {
  'DESCRIPTOR' : _CAMERAKEYPOINT,
  '__module__' : 'waymo_open_dataset.protos.keypoint_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.keypoints.CameraKeypoint)
  })
_sym_db.RegisterMessage(CameraKeypoint)

CameraKeypoints = _reflection.GeneratedProtocolMessageType('CameraKeypoints', (_message.Message,), {
  'DESCRIPTOR' : _CAMERAKEYPOINTS,
  '__module__' : 'waymo_open_dataset.protos.keypoint_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.keypoints.CameraKeypoints)
  })
_sym_db.RegisterMessage(CameraKeypoints)

LaserKeypoint = _reflection.GeneratedProtocolMessageType('LaserKeypoint', (_message.Message,), {
  'DESCRIPTOR' : _LASERKEYPOINT,
  '__module__' : 'waymo_open_dataset.protos.keypoint_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.keypoints.LaserKeypoint)
  })
_sym_db.RegisterMessage(LaserKeypoint)

LaserKeypoints = _reflection.GeneratedProtocolMessageType('LaserKeypoints', (_message.Message,), {
  'DESCRIPTOR' : _LASERKEYPOINTS,
  '__module__' : 'waymo_open_dataset.protos.keypoint_pb2'
  # @@protoc_insertion_point(class_scope:waymo.open_dataset.keypoints.LaserKeypoints)
  })
_sym_db.RegisterMessage(LaserKeypoints)


# @@protoc_insertion_point(module_scope)
