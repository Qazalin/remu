
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_2n2>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001608: BF89FC07
	s_load_b64 s[2:3], s[2:3], null                            // 00000000160C: F4040081 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001614: BF89FC07
	v_add_f32_e64 v0, s2, 0                                    // 000000001618: D5030000 00010002
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001620: BF870001
	v_dual_mov_b32 v1, 0 :: v_dual_add_f32 v0, s3, v0          // 000000001624: CA080080 01000003
	global_store_b32 v1, v0, s[0:1]                            // 00000000162C: DC6A0000 00000001
	s_nop 0                                                    // 000000001634: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001638: BFB60003
	s_endpgm                                                   // 00000000163C: BFB00000
