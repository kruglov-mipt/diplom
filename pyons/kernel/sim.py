# #
# # File: sim.py
# #
# import types
# from enum import Enum
#
# import pyons.kernel.queues as queues
#
#
# #######################################################################
# # KERNEL
# #
# # Simulation kernel is a singleton that runs the simulation.
# # TODO: write normal reference
# #######################################################################
#
# class Kernel(object, metaclass=Singleton):
#     """
#     Simulation kernel.
#     """
#
#     class Context(object):
#         """
#         Object groups data about current running managed functions, e.g.
#         handlers, initializers, etc.
#
#         Now only used to track handlers to automatically determine the
#         source of the event. In future will be used for manual execution
#         and other features.
#         """
#         class FType(Enum):
#             INITIALIZER = 0
#             HANDLER = 1
#             FINALIZER = 2
#
#         def __init__(self):
#             self.call_stack = []  # (entity [=None for static], func)
#
#         def push(self, func, entity=None, ft=Kernel.Context.FType.HANDLER):
#             self.call_stack.append((entity, func, ft))
#
#         def pop(self):
#             return self.call_stack.pop()
#
#         @property
#         def function(self):
#             return self.call_stack[-1][1] if self.call_stack else None
#
#         @property
#         def entity(self):
#             return self.call_stack[-1][0] if self.call_stack else None
#
#         @property
#         def function_type(self):
#             return self.call_stack[-1][2] if self.call_stack else None
#
#         def reset(self):
#             self.call_stack = []
#
#     class Envelope(object):
#         """
#         Event envelope. Contains information about a sender and target
#         entities and handlers.
#
#         When a kernel needs to enqueue an event, it creates an envelope
#         and enqueues it rather then the event itself.
#         """
#         def __init__(self):
#             self.fire_time = None
#             self.source_entity = None
#             self.source_handler = None
#             self.target_entity = None
#             self.target_handler = None
#             self.event = None
#
#     class State(Enum):
#         IDLE = 0
#         INITIALIZATION = 1
#         RUNNING = 2
#         FINALIZATION = 3
#         FINISHED = 4
#
#     def __init__(self):
#         self._queue = None            # delayed events - sent with schedule()
#         self._immediate_events = []     # immediate events - sent with send()
#         self._context = Kernel.Context()               # who is executing now
#         self._entities = []           # entities registered within the kernel
#         self._new_entities = []       # entities to be initialized
#         self._state = Kernel.State.IDLE
#         self._events_num = 0
#         self._stop_flag = False
#         self._aborted = False
#         self.model = None
#         self._environment = None
#         self._time = 0.0
#
#     @property
#     def state(self):
#         return self._state
#
#     @property
#     def environment(self):
#         if self._environment is None:
#             self.environment = Environment()
#         return self._environment
#
#     @environment.setter
#     def environment(self, env):
#         self._environment = env
#         env.kernel = self
#
#     @property
#     def events(self):
#         return None     # TODO
#
#     @property
#     def time(self):
#         return self._time
#
#     @property
#     def context(self):
#         return self._context
#
#     @property
#     def num_events_handled(self):
#         return self._events_num
#
#     def run(self, mode=None, max_events=None, a_queue=None):
#         """
#         Run the simulation. This includes:
#
#         1) initialization
#
#         2) event loop iteration
#
#         3) finalization.
#
#         At the first step, all the methods registered with
#         @static_initializer() decorator would be called.
#         These methods are expected to create initial events
#         and prepare the model.
#
#         The second step (running the event loop) is performed
#         till event queue is not empty. User can also stop
#         the loop with stop() or abort() call. If ``max_events``
#         argument was set, it specifies the number of events
#         after which the loop will stop.
#
#         The third step (finalization) is performed if the second
#         step ended with anything except abort() call. It
#         calls all the methods registered with @finalizer decorator.
#
#         Args:
#             mode: not supported now
#
#             max_events: after handling this number of
#             events simulation will stop
#
#             a_queue: the event queue. By default,
#             ``queues.HeapQueue`` would be used.
#         """
#         if self._state not in [Kernel.State.IDLE, Kernel.State.FINISHED]:
#             return
#
#         self._immediate_events = []
#         self._queue = a_queue if a_queue is not None else queues.HeapQueue(
#             event_time_getter=lambda envelope: envelope.fire_time)
#
#         self._state = Kernel.State.IDLE
#         self._events_num = 0
#         self._stop_flag = False
#         self._aborted = False
#         self._time = 0.0
#         self.context.reset()
#         self._initialize()
#
#         #
#         # Main event-loop
#         #
#         self.context.initializing = False
#         while not self._stop_flag and not self._queue.empty:
#             if max_events is not None and self._events_num > max_events:
#                 self._stop_flag = True
#             dead_entities = self._list_dead_entities()
#             for ent in dead_entities:
#                 self.detach_entity(ent)
#             self._stop_flag = self._test_stop_conditions()
#             if not self._stop_flag:
#                 e = self._queue.pop()
#                 self._time = e.fire_time
#                 self._dispatch(e)
#                 self._events_num += 1
#
#         if not self._aborted:
#             self._finalize()                    # finalize the simulation
#
#     def stop(self):
#         """
#         Stop the simulation. After the control will come back
#         to the event-loop, no more events would be taken
#         and finalization will take place.
#         """
#         self._stop_flag = True
#
#     def abort(self):
#         """
#         Stop the simulation and don't perform finalization.
#         """
#         self._stop_flag = True
#         self._aborted = True
#
#     def attach_entity(self, entity):
#         if type(entity) is not Managed:
#             raise RuntimeError("entity must have Managed metaclass")
#         self._entities.append(entity)
#         if self.state in [Kernel.State.INITIALIZATION, Kernel.State.IDLE]:
#             self._new_entities.append(entity)
#         elif self.state in [Kernel.State.FINALIZATION, Kernel.State.FINISHED]:
#             raise RuntimeWarning("entity can not be attached when finished")
#         else:
#             self._init_entity(entity)
#             self._entities.append(entity)
#
#     def detach_entity(self, entity, aborted=False):
#         if entity in self._new_entities:
#             self._new_entities.remove(entity)
#         if entity in self._entities:
#             if not aborted:
#                 self._finalize_entity(entity)
#             self._entities.remove(entity)
#
#             def entity_envelope(envelope):
#                 return envelope.source_entity is entity or \
#                        envelope.target_entity is entity
#
#             self._queue.remove(predicate=entity_envelope)
#             self._immediate_events = filter(entity_envelope,
#                                             self._immediate_events)
#
#     def schedule_event(self, event, fire_time=None, target_entity=None,
#                        target_handler=None):
#         ft = self.context.function_type
#         if ft is None:
#             raise RuntimeError("scheduling disallowed outside managed context")
#         elif ft is Kernel.Context.FType.FINALIZER:
#             raise RuntimeError("scheduling disallowed during finalization")
#         envelope = Kernel.Envelope()
#         envelope.source_entity = self.context.entity
#         envelope.source_handler = self.context.function
#         envelope.target_entity = target_entity
#         envelope.target_handler = target_handler
#         envelope.fire_time = fire_time
#         envelope.event = event
#         if fire_time is not None:
#             return self._queue.push(envelope)
#         else:
#             self._immediate_events.append((self._queue.next_index(), envelope))
#
#     def cancel_event(self, event_index):
#         if self._queue.has_index(event_index):
#             self._queue.remove(index=event_index)
#         else:
#             for i in range(0, len(self._immediate_events)):
#                 ind, ev = self._immediate_events[i]
#                 if ind == event_index:
#                     self._immediate_events.remove((ind, ev))
#                     break
#
#     def _init_entity(self, entity):
#         if type(entity) is
#
#     def _finalize_entity(self, entity):
#         pass
#
#     def _list_dead_entities(self):
#         pass
#
#     def _test_stop_conditions(self):
#         pass
#
#     def _initialize(self):
#         for stage, fn in Kernel.staged_functions_iterator(self.__initializers):
#             if self.environment is not None:
#                 self.environment.debug("[init stage='{stage}'] calling {fn}".format(fn=fn.sim_name, stage=stage),
#                                        sender='kernel')
#             fn(model=self.model)
#
#     def _finalize(self):
#         for stage, fn in Kernel.staged_functions_iterator(self.__finalizers):
#             if self.environment is not None:
#                 self.environment.debug("[fin stage='{stage}'] calling {fn}".format(fn=fn.sim_name, stage=stage),
#                                        sender='kernel')
#             fn(model=self.model)
#
#     def _dispatch(self, envelope):
#         handler = self.__default_handler
#         if envelope.target_handler is not None:
#             handler = envelope.target_handler
#         else:
#             for condition, cond_handler in self.__conditional_handlers:
#                 if condition(envelope.event):
#                     handler = cond_handler
#                     break
#
#         if handler is None:
#             raise RuntimeError("handler not found for the event '{}'".format(envelope.event))
#
#         if self.environment is not None:
#             self.environment.debug("[>>>] calling {fn}() for event='{event}'".format(
#                 fn=handler.sim_name, event=envelope.event), sender='kernel')
#
#         handler(envelope.event, model=self.model)
#
#
# class LogLevel(Enum):
#     ALL = -1
#     TRACE = 0
#     DEBUG = 1
#     FINE = 2
#     INFO = 3
#     WARNING = 4
#     ERROR = 5
#     OFF = 6
#
#
# class Environment(object):
#     def __init__(self):
#         self.kernel = None
#         self.log_level = LogLevel.INFO
#         self.time_precision = 9
#         self.sender_field_width = 16
#
#     def log(self, level, message, sender=None):
#         assert LogLevel.TRACE.value <= level.value <= LogLevel.ERROR.value
#         if level.value >= self.log_level.value:
#             s_time = "{time:0{width}.{precision}f} ".format(time=self.kernel.time, width=self.time_precision+6,
#                                                             precision=self.time_precision)
#             s_level = "[{level.name:^7s}] ".format(level=level)
#             if sender is not None:
#                 s_sender = "({sender:^{width}s}) ".format(sender=sender, width=self.sender_field_width)
#             else:
#                 s_sender = ""
#             print("{time}{level}{sender}{message}".format(time=s_time, level=s_level, sender=s_sender, message=message))
#
#     def trace_enter_function(self, fn, indention_level=0):
#         self.log(LogLevel.TRACE, "{indent}----> {fun_name}".format(
#             indent="  "*indention_level, fun_name=fn.__name__))
#
#     def trace_exit_function(self, fn, indention_level=0):
#         self.log(LogLevel.TRACE, "{indent}<---- {fun_name}".format(
#             indent="  "*indention_level, fun_name=fn.__name__))
#
#     def debug(self, message, sender=None):
#         self.log(LogLevel.DEBUG, message, sender)
#
#     def info(self, message, sender=None):
#         self.log(LogLevel.INFO, message, sender)
#
#     def fine(self, message, sender=None):
#         self.log(LogLevel.FINE, message, sender)
#
#     def warning(self, message, sender=None):
#         self.log(LogLevel.WARNING, message, sender)
#
#     def error(self, message, sender=None):
#         self.log(LogLevel.ERROR, message, sender)
#         self.kernel.abort()
#
#
# def now():
#     return Kernel().time
#
#
# def run(model=None, mode=None, max_events=None, queue=None):
#     kernel = Kernel()
#     kernel.model = model
#     kernel.run(mode=mode, max_events=max_events, a_queue=queue)
#
#
# def stop():
#     Kernel().stop()
#
#
# def abort():
#     Kernel().abort()
#
#
# def schedule(event, dt, handler=None):
#     kernel = Kernel()
#     return kernel.schedule(event, fire_time=kernel.time + dt, target_handler=handler)
#
#
# def create_timeout(event, dt):
#     kernel = Kernel()
#     return kernel.schedule(event, fire_time=kernel.time + dt, target_handler=kernel.context.handler)
#
#
# def send_event(event, handler=None):
#     kernel = Kernel()
#     return kernel.schedule(event, fire_time=kernel.time, target_handler=handler)
#
#
# def cancel_event(event_index):
#     Kernel().cancel(event_index=event_index)
#
#
# def debug(message, sender=None):
#     Kernel().environment.debug(message, sender=sender)
#
#
# def fine(message, sender=None):
#     Kernel().environment.fine(message, sender=sender)
#
#
# def info(message, sender=None):
#     Kernel().environment.info(message, sender=sender)
#
#
# def warning(message, sender=None):
#     Kernel().environment.warning(message, sender=sender)
#
#
# def error(message, sender=None):
#     Kernel().environment.error(message, sender=sender)
#
#
# def get_log_level():
#     return Kernel().environment.log_level
#
#
# def setup_env(log_level=LogLevel.DEBUG, sender_field_width=16, time_precision=9):
#     env = Kernel().environment
#     env.log_level = log_level
#     env.sender_field_width = sender_field_width
#     env.time_precision = time_precision
#
