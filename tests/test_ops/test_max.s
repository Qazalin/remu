
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_135>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	v_mov_b32_e32 v0, 0xff800000                               // 000000001608: 7E0002FF FF800000
	s_mov_b64 s[4:5], 0                                        // 000000001610: BE840180
	s_set_inst_prefetch_distance 0x1                           // 000000001614: BF840001
	s_nop 0                                                    // 000000001618: BF800000
	s_nop 0                                                    // 00000000161C: BF800000
	s_nop 0                                                    // 000000001620: BF800000
	s_nop 0                                                    // 000000001624: BF800000
	s_nop 0                                                    // 000000001628: BF800000
	s_nop 0                                                    // 00000000162C: BF800000
	s_nop 0                                                    // 000000001630: BF800000
	s_nop 0                                                    // 000000001634: BF800000
	s_nop 0                                                    // 000000001638: BF800000
	s_nop 0                                                    // 00000000163C: BF800000
	s_waitcnt lgkmcnt(0)                                       // 000000001640: BF89FC07
	s_add_u32 s6, s2, s4                                       // 000000001644: 80060402
	s_addc_u32 s7, s3, s5                                      // 000000001648: 82070503
	v_max_f32_e32 v0, v0, v0                                   // 00000000164C: 20000100
	s_clause 0x1                                               // 000000001650: BF850001
	s_load_b256 s[8:15], s[6:7], null                          // 000000001654: F40C0203 F8000000
	s_load_b128 s[16:19], s[6:7], 0x20                         // 00000000165C: F4080403 F8000020
	s_add_u32 s4, s4, 60                                       // 000000001664: 8004BC04
	s_addc_u32 s5, s5, 0                                       // 000000001668: 82058005
	s_cmpk_eq_i32 s4, 0x21c                                    // 00000000166C: B184021C
	s_waitcnt lgkmcnt(0)                                       // 000000001670: BF89FC07
	v_max_f32_e64 v1, s8, s8                                   // 000000001674: D5100001 00001008
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000167C: BF870091
	v_max_f32_e32 v0, v1, v0                                   // 000000001680: 20000101
	v_max3_f32 v0, s10, s9, v0                                 // 000000001684: D61C0000 0400120A
	s_clause 0x1                                               // 00000000168C: BF850001
	s_load_b64 s[8:9], s[6:7], 0x30                            // 000000001690: F4040203 F8000030
	s_load_b32 s6, s[6:7], 0x38                                // 000000001698: F4000183 F8000038
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016A0: BF870091
	v_max3_f32 v0, s12, s11, v0                                // 0000000016A4: D61C0000 0400160C
	v_max3_f32 v0, s14, s13, v0                                // 0000000016AC: D61C0000 04001A0E
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016B4: BF870091
	v_max3_f32 v0, s16, s15, v0                                // 0000000016B8: D61C0000 04001E10
	v_max3_f32 v0, s18, s17, v0                                // 0000000016C0: D61C0000 04002212
	s_waitcnt lgkmcnt(0)                                       // 0000000016C8: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016CC: BF870091
	v_max3_f32 v0, s8, s19, v0                                 // 0000000016D0: D61C0000 04002608
	v_max3_f32 v0, s6, s9, v0                                  // 0000000016D8: D61C0000 04001206
	s_cbranch_scc0 65495                                       // 0000000016E0: BFA1FFD7 <r_135+0x40>
	s_set_inst_prefetch_distance 0x2                           // 0000000016E4: BF840002
	v_mov_b32_e32 v1, 0                                        // 0000000016E8: 7E020280
	global_store_b32 v1, v0, s[0:1]                            // 0000000016EC: DC6A0000 00000001
	s_nop 0                                                    // 0000000016F4: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016F8: BFB60003
	s_endpgm                                                   // 0000000016FC: BFB00000
