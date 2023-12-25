
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_9>:
	s_load_b128 s[8:11], s[0:1], null                          // 000000001600: F4080200 F8000000
	v_mov_b32_e32 v1, 0                                        // 000000001608: 7E020280
	s_waitcnt lgkmcnt(0)                                       // 00000000160C: BF89FC07
	s_load_b256 s[0:7], s[10:11], null                         // 000000001610: F40C0005 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	v_max_f32_e64 v0, -s0, -s0                                 // 00000000161C: D5100000 60000000
	s_load_b32 s0, s[10:11], 0x20                              // 000000001624: F4000005 F8000020
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000162C: BF870091
	v_max_f32_e32 v0, 0xff800000, v0                           // 000000001630: 200000FF FF800000
	v_max3_f32 v0, -s2, -s1, v0                                // 000000001638: D61C0000 64000202
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001640: BF870091
	v_max3_f32 v0, -s4, -s3, v0                                // 000000001644: D61C0000 64000604
	v_max3_f32 v0, -s6, -s5, v0                                // 00000000164C: D61C0000 64000A06
	s_waitcnt lgkmcnt(0)                                       // 000000001654: BF89FC07
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001658: BF870001
	v_min3_f32 v0, s0, s7, -v0                                 // 00000000165C: D6190000 84000E00
	global_store_b32 v1, v0, s[8:9]                            // 000000001664: DC6A0000 00080001
	s_nop 0                                                    // 00000000166C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001670: BFB60003
	s_endpgm                                                   // 000000001674: BFB00000
