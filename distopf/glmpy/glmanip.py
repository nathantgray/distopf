"""
Author: Pacific Northwest National Laboratory (PNNL)
Edits by Nathan Gray (WSU) 2021-2022 to add support for pathlib.Path

This program reads GridLAB-D Model files (.glm) into python dictionaries
and writes python dictionaries to a single GridLAB-D model file.
"""

import re
from pathlib import Path


def read(fn, basedir=None, buf=None):
    """
    Recursive glm line reader:
        Appends uncommented glm lines in fh and dependencies to buf
    Inputs:
        fn: file path
        basedir: base model directory
        buf: line buffer (list of glm lines as strings)
    """
    fn = Path(fn)
    if basedir is None:
        basedir = fn.parent
    else:
        basedir = Path(basedir)
    if buf is None:
        buf = []
    print(f"...Reading {fn}")
    with open(fn, "r") as fh:
        line = fh.readline()
        line = remove_comment_inline(line)  # remove comments (anything after //...)
        while line != "":  # Skip Blank lines
            while re.match("\s*//", line) or re.match(
                "\s+$", line
            ):  # Skip comments and whitespace-only lines
                line = fh.readline()
            line = remove_comment_inline(line)  # remove comments (anything after //...)
            m = re.search("#\s?include\s+['\"]?([^\.]+\.glm)", line)
            if m:
                # Found dependency
                dep_fn = Path(m.group(1))
                read(basedir / dep_fn, basedir, buf)
            else:
                line = remove_comment_inline(
                    line
                )  # remove comments (anything after //...)
                buf.append(line)
            line = fh.readline()
            line = remove_comment_inline(line)  # remove comments (anything after //...)
    return buf


def write(ofn, model, clock, directives, modules, classes, schedules):
    # Open the output file
    with open(ofn, "w") as outf:
        # Write the '#' directives
        for directive in directives:
            outf.write(directive + "\n")
        outf.write("\n")

        # Write the clock, if found
        if len(clock) > 0:
            outf.write("clock {\n")
            for param in clock:
                outf.write("\t" + param + " " + clock[param] + ";\n")
            outf.write("}\n")
        outf.write("\n")

        # Write the modules
        for module in modules:
            outf.write("module " + module)
            if len(modules[module]) == 0:
                outf.write(";\n")
            else:
                outf.write(" {\n")
                for param in modules[module]:
                    outf.write("\t" + param + " " + modules[module][param] + ";\n")
                outf.write("};\n")
            outf.write("\n")

        # Write the classes
        for c in classes:
            outf.write("class " + c)
            if len(classes[c]) == 0:
                outf.write(";\n")
            else:
                outf.write(" {\n")
                for param in classes[c]:
                    outf.write("\t" + param + " " + classes[c][param] + ";\n")
                outf.write("};\n")
            outf.write("\n")

        # Write the schedules
        for s in schedules:
            for sline in schedules[s]:
                outf.write(sline)
            outf.write("\n")

        # Write the objects
        objctr = 0
        for t in model:
            for o in model[t]:
                objctr += 1
                # write each object
                outf.write("object " + t + " {\n")
                outf.write("\tname " + o + ";\n")
                for p in model[t][o]:
                    try:
                        outf.write(f"\t{p} {model[t][o][p]};\n")
                    except TypeError:
                        raise TypeError(f"Failed to concatenate: {p} {model[t][o][p]}")
                outf.write("}\n\n")
        print("\tTotal objects written: " + str(objctr))


def obj(parent, model, line, itr, oidh, octr):
    """
    Store an object in the model structure
    Inputs:
        parent: name of parent object (used for nested object defs)
        model: dictionary model structure
        line: glm line containing the object definition
        itr: iterator over the list of lines
        oidh: hash of object id's to object names
        octr: object counter
    """
    octr += 1
    # Identify the object type
    m = re.search("object ([^:{\s]+)[:{\s]", line, re.IGNORECASE)

    _type = m.group(1)
    # If the object has an id number, store it
    n = re.search("object ([^:]+:[^{\s]+)", line, re.IGNORECASE)
    if n:
        oid = n.group(1)
    line = next(itr)
    # Collect parameters
    oend = 0
    oname = None
    params = {}
    if parent is not None:
        params["parent"] = parent
    # print('nested '+type)
    while not oend:
        m = re.match("\s*(\S+)\s+([^;{]+)[;{]", line)
        if m:
            if re.search("\$", line):  # check if value is a defined variable
                # redo match but don't exclude { in the match and don't use it to end the match
                m = re.search("\s*(\S+)\s+([^;]+)[;]", line)
            # found a parameter
            param = m.group(1)
            # check for quotations in the value
            mm = re.search('("[^"]*")', line)
            if mm:
                val = mm.group(1)
            else:
                val = m.group(2)
            intobj = 0
            if param == "name":
                oname = val
            elif param == "object":
                # found a nested object
                intobj += 1
                if oname is None:
                    # print('ERROR: nested object defined before parent name')
                    # quit()
                    raise RuntimeError("nested object defined before parent name")
                line, octr = obj(oname, model, line, itr, oidh, octr)
            elif re.match("object", val):
                # found an inline object
                intobj += 1
                line, octr = obj(None, model, line, itr, oidh, octr)
                params[param] = "OBJECT_" + str(octr)
            else:
                params[param] = val
        if re.search("}", line) and not re.search("\$", line):
            if intobj:
                intobj -= 1
                line = next(itr)
                line = remove_comment_inline(
                    line
                )  # remove comments (anything after //...)
            else:
                oend = 1
        else:
            line = next(itr)
            line = remove_comment_inline(line)  # remove comments (anything after //...)
    # If undefined, use a default name
    if oname is None:
        oname = "OBJECT_" + str(octr)
    oidh[oname] = oname
    # Hash an object identifier to the object name
    if n:
        oidh[oid] = oname
    # Add the object to the model
    if _type not in model:
        # New object type
        model[_type] = {}
    model[_type][oname] = {}
    for param in params:
        model[_type][oname][param] = params[param]
    # Return the
    return line, octr


def parse(lines):
    # --------------------------
    # Build the model structure
    # --------------------------
    model = {}
    clock = {}
    modules = {}
    classes = {}
    schedules = {}
    directives = []
    octr = 0
    h = {}  # OID hash
    itr = iter(lines)
    for line in itr:
        # Look for objects
        if re.search("object", line):
            line, octr = obj(None, model, line, itr, h, octr)
        # Look for # directives
        if re.match("#\s?\w", line):
            if re.match("#/s?if", line, re.IGNORECASE):
                print("WARNING: #if directive is not supported by glm")
            elif re.match("#/s?endif", line, re.IGNORECASE):
                print("WARNING: #endif directive is not supported by glm")
            else:
                directives.append(line)
        # Look for the clock
        m_clock = re.match("clock\s*([;{])", line, re.IGNORECASE)
        if m_clock:
            # Clock found: look for parameters
            if m_clock.group(1) == "{":
                # multi-line clock definition
                oend = 0
                while not oend:
                    line = next(itr)
                    m_param = re.search("(\w+)\s+([^;\n]+)", line)
                    if m_param:
                        # Parameter found
                        clock[m_param.group(1)] = m_param.group(2)
                    if re.search("}", line):
                        # End of the clock definition
                        oend = 1
        # Look for module defintions
        m_mtype = re.search("module\s+(\w+)\s*([;{])", line, re.IGNORECASE)
        if m_mtype:
            # Module found: look for parameters
            modules[m_mtype.group(1)] = {}
            if m_mtype.group(2) == "{":
                # multi-line module definition
                oend = 0
                while not oend:
                    line = next(itr)
                    m_param = re.search("(\w+)\s+([^;\n]+)", line)
                    if m_param:
                        # Parameter found
                        modules[m_mtype.group(1)][m_param.group(1)] = m_param.group(2)
                    if re.search("}", line):
                        # End of the module
                        oend = 1
        # Look for class definitions
        m_ctype = re.search("class\W+(\w+)\s*([;{])", line, re.IGNORECASE)
        if m_ctype:
            # Class found: look for parameters
            classes[m_ctype.group(1)] = {}
            if m_ctype.group(2) == "{":
                # multi-line class definition
                oend = 0
                while not oend:
                    line = next(itr)
                    m_param = re.search("(\w+)\s+([^;\n]+)", line)
                    if m_param:
                        # Parameter found
                        classes[m_ctype.group(1)][m_param.group(1)] = m_param.group(2)
                    if re.search("}", line):
                        # End of the class
                        oend = 1
        # Look for schedule definitions
        m_sched = re.search("schedule\W+(\w+)\s*([;{])", line, re.IGNORECASE)
        if m_sched:
            # schedule found
            schedules[m_sched.group(1)] = []
            schedules[m_sched.group(1)].append(line)
            if m_sched.group(2) == "{":
                # multi-line schedule
                oend = 0
                while not oend:
                    line = next(itr)
                    schedules[m_sched.group(1)].append(line)
                    if re.search("}", line):
                        # end of the schedule
                        oend = 1

    # ------------------------------
    # Update all object name values
    # ------------------------------
    for t in model:
        for o in model[t]:
            for p in model[t][o]:
                if model[t][o][p] in h:
                    model[t][o][p] = h[model[t][o][p]]

    # -------
    # Return
    # -------
    return model, clock, directives, modules, classes, schedules


def remove_comment_inline(line):
    line_w_comment = re.search("(.+)(//.*$)", line)
    if line_w_comment:
        line = line_w_comment.group(0)  # remove comments (anything after //...)
        if "//" in line:
            line = line_w_comment.group(1)
    return line
