
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_100_10>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	v_dual_mov_b32 v0, 0xff800000 :: v_dual_mov_b32 v1, 0      // 000000001610: CA1000FF 00000080 FF800000
	s_lshl_b64 s[4:5], s[4:5], 2                               // 00000000161C: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001620: BF89FC07
	s_add_u32 s2, s2, s4                                       // 000000001624: 80020402
	s_addc_u32 s3, s3, s5                                      // 000000001628: 82030503
	s_add_u32 s0, s0, s4                                       // 00000000162C: 80000400
	s_clause 0x9                                               // 000000001630: BF850009
	s_load_b32 s6, s[2:3], null                                // 000000001634: F4000181 F8000000
	s_load_b32 s7, s[2:3], 0x190                               // 00000000163C: F40001C1 F8000190
	s_load_b32 s8, s[2:3], 0x320                               // 000000001644: F4000201 F8000320
	s_load_b32 s9, s[2:3], 0x4b0                               // 00000000164C: F4000241 F80004B0
	s_load_b32 s10, s[2:3], 0x640                              // 000000001654: F4000281 F8000640
	s_load_b32 s11, s[2:3], 0x7d0                              // 00000000165C: F40002C1 F80007D0
	s_load_b32 s12, s[2:3], 0x960                              // 000000001664: F4000301 F8000960
	s_load_b32 s13, s[2:3], 0xaf0                              // 00000000166C: F4000341 F8000AF0
	s_load_b32 s14, s[2:3], 0xc80                              // 000000001674: F4000381 F8000C80
	s_load_b32 s2, s[2:3], 0xe10                               // 00000000167C: F4000081 F8000E10
	s_addc_u32 s1, s1, s5                                      // 000000001684: 82010501
	s_waitcnt lgkmcnt(0)                                       // 000000001688: BF89FC07
	v_max3_f32 v0, s7, s6, v0                                  // 00000000168C: D61C0000 04000C07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001694: BF870091
	v_max3_f32 v0, s9, s8, v0                                  // 000000001698: D61C0000 04001009
	v_max3_f32 v0, s11, s10, v0                                // 0000000016A0: D61C0000 0400140B
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016A8: BF870091
	v_max3_f32 v0, s13, s12, v0                                // 0000000016AC: D61C0000 0400180D
	v_max3_f32 v0, s2, s14, v0                                 // 0000000016B4: D61C0000 04001C02
	global_store_b32 v1, v0, s[0:1]                            // 0000000016BC: DC6A0000 00000001
	s_nop 0                                                    // 0000000016C4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016C8: BFB60003
	s_endpgm                                                   // 0000000016CC: BFB00000
