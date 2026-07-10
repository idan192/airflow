# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import annotations

import heapq
from collections import deque
from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from airflow_shared.dagnode.node import DagProtocol


class _TaskNode(Protocol):
    task_id: str
    upstream_task_ids: set[str]


class _VisibleNode(Protocol):
    @property
    def node_id(self) -> str | None: ...


VisibleNodeT = TypeVar("VisibleNodeT", bound=_VisibleNode)


def _ensure_task_graph_is_acyclic(
    tasks: tuple[_TaskNode, ...], dag_id: str, cycle_exception: type[Exception]
) -> None:
    task_ids = {task.task_id for task in tasks}
    in_degree = {task_id: 0 for task_id in task_ids}
    successors: dict[str, list[str]] = {task_id: [] for task_id in task_ids}
    for task in tasks:
        for upstream_task_id in task.upstream_task_ids & task_ids:
            in_degree[task.task_id] += 1
            successors[upstream_task_id].append(task.task_id)

    ready = deque(task_id for task_id, degree in in_degree.items() if degree == 0)
    processed = 0
    while ready:
        task_id = ready.popleft()
        processed += 1
        for successor in successors[task_id]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                ready.append(successor)
    if processed != len(task_ids):
        raise cycle_exception(f"A cyclic dependency occurred in dag: {dag_id}")


def _find_strongly_connected_components(
    successors: list[list[int]], dependencies: list[tuple[int, ...]]
) -> list[list[int]]:
    visited = bytearray(len(successors))
    finish_order: list[int] = []
    for start in range(len(successors)):
        if visited[start]:
            continue
        visited[start] = 1
        dfs_stack = [(start, 0)]
        while dfs_stack:
            node, successor_position = dfs_stack[-1]
            if successor_position < len(successors[node]):
                successor = successors[node][successor_position]
                dfs_stack[-1] = (node, successor_position + 1)
                if not visited[successor]:
                    visited[successor] = 1
                    dfs_stack.append((successor, 0))
                continue
            finish_order.append(node)
            dfs_stack.pop()

    assigned = bytearray(len(successors))
    components: list[list[int]] = []
    for start in reversed(finish_order):
        if assigned[start]:
            continue
        assigned[start] = 1
        component: list[int] = []
        reverse_stack = [start]
        while reverse_stack:
            node = reverse_stack.pop()
            component.append(node)
            for predecessor in dependencies[node]:
                if not assigned[predecessor]:
                    assigned[predecessor] = 1
                    reverse_stack.append(predecessor)
        components.append(component)
    return components


def resolve_task_group_projection_cycle(
    *,
    dag: DagProtocol,
    tasks: tuple[_TaskNode, ...],
    nodes: list[VisibleNodeT],
    projected: list[tuple[int, ...]],
    cycle_exception: type[Exception],
) -> list[VisibleNodeT]:
    """Order a cyclic visible-node projection without masking a real task cycle."""
    _ensure_task_graph_is_acyclic(tasks, dag.dag_id, cycle_exception)

    successors: list[list[int]] = [[] for _ in nodes]
    for child_position, dependencies in enumerate(projected):
        for dependency in dependencies:
            successors[dependency].append(child_position)

    components = _find_strongly_connected_components(successors, projected)
    component_of = {
        node_position: component_position
        for component_position, component in enumerate(components)
        for node_position in component
    }
    component_dependencies: list[set[int]] = [set() for _ in components]
    component_successors: list[set[int]] = [set() for _ in components]
    for node_position, dependencies in enumerate(projected):
        node_component = component_of[node_position]
        for dependency in dependencies:
            dependency_component = component_of[dependency]
            if dependency_component == node_component:
                continue
            component_dependencies[node_component].add(dependency_component)
            component_successors[dependency_component].add(node_component)

    component_keys = [
        min((position, str(nodes[position].node_id)) for position in component) for component in components
    ]
    ready = [
        (*component_keys[position], position)
        for position, dependencies in enumerate(component_dependencies)
        if not dependencies
    ]
    heapq.heapify(ready)
    order: list[VisibleNodeT] = []
    while ready:
        _, _, component_position = heapq.heappop(ready)
        order.extend(nodes[position] for position in sorted(components[component_position]))
        for successor in component_successors[component_position]:
            component_dependencies[successor].discard(component_position)
            if not component_dependencies[successor]:
                heapq.heappush(ready, (*component_keys[successor], successor))
    return order
