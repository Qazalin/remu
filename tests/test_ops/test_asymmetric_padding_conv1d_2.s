
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_4_2>:
	s_clause 0x1                                               // 000000001600: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001604: F4080100 F8000000
	s_load_b64 s[2:3], s[0:1], 0x10                            // 00000000160C: F4040080 F8000010
	s_mov_b32 s8, s15                                          // 000000001614: BE88000F
	s_ashr_i32 s9, s15, 31                                     // 000000001618: 86099F0F
	s_add_i32 s10, s15, -1                                     // 00000000161C: 810AC10F
	s_lshl_b64 s[0:1], s[8:9], 2                               // 000000001620: 84808208
	s_mov_b32 s9, 0                                            // 000000001624: BE890080
	s_waitcnt lgkmcnt(0)                                       // 000000001628: BF89FC07
	s_add_u32 s6, s6, s0                                       // 00000000162C: 80060006
	s_addc_u32 s7, s7, s1                                      // 000000001630: 82070107
	s_add_u32 s6, s6, -8                                       // 000000001634: 8006C806
	s_addc_u32 s7, s7, -1                                      // 000000001638: 8207C107
	s_cmp_lt_i32 s10, 1                                        // 00000000163C: BF04810A
	s_mov_b32 s10, 0                                           // 000000001640: BE8A0080
	s_cbranch_scc1 2                                           // 000000001644: BFA20002 <r_4_2+0x50>
	s_load_b32 s10, s[6:7], null                               // 000000001648: F4000283 F8000000
	s_load_b32 s11, s[2:3], null                               // 000000001650: F40002C1 F8000000
	s_cmp_lt_i32 s8, 1                                         // 000000001658: BF048108
	s_cbranch_scc1 2                                           // 00000000165C: BFA20002 <r_4_2+0x68>
	s_load_b32 s9, s[6:7], 0x4                                 // 000000001660: F4000243 F8000004
	s_load_b32 s2, s[2:3], 0x4                                 // 000000001668: F4000081 F8000004
	s_waitcnt lgkmcnt(0)                                       // 000000001670: BF89FC07
	v_fma_f32 v0, s10, s11, 0                                  // 000000001674: D6130000 0200160A
	s_add_u32 s0, s4, s0                                       // 00000000167C: 80000004
	s_addc_u32 s1, s5, s1                                      // 000000001680: 82010105
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001684: BF870091
	v_fmac_f32_e64 v0, s9, s2                                  // 000000001688: D52B0000 00000409
	v_dual_mov_b32 v1, 0 :: v_dual_max_f32 v0, 0, v0           // 000000001690: CA140080 01000080
	global_store_b32 v1, v0, s[0:1]                            // 000000001698: DC6A0000 00000001
	s_nop 0                                                    // 0000000016A0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016A4: BFB60003
	s_endpgm                                                   // 0000000016A8: BFB00000
